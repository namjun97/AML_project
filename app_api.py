r"""FastAPI 백엔드 — AML 탐지 + SAR 생성 REST API + OpenAI 호환 엔드포인트.

Streamlit(app.py)이 프론트+백엔드를 한 덩어리로 처리하던 것을, 탐지·보고서
파이프라인을 그대로 재사용하면서 **백엔드 서비스**로 분리한 계층이다.
(로직은 loaders/·reporters/·models/ 에 이미 있고, 이 파일은 그 위의 얇은 API 층.)

제공 엔드포인트
  REST (자체 프론트엔드/통합용)
    GET  /health
    GET  /api/accounts/watchlist?limit=20   - 위험도 상위 계좌 목록
    GET  /api/accounts/{node_idx}           - 단일 계좌 탐지 결과(+SHAP 근거)
    POST /api/sar/generate {node_idx}       - LangGraph 파이프라인으로 SAR 생성
  OpenAI 호환 (OpenWebUI 프론트엔드 연동용 — 프론트 코드 0줄)
    GET  /v1/models
    POST /v1/chat/completions               - 메시지에서 계좌 id 파싱 -> SAR 스트리밍

실행:
    .\.venv\Scripts\uvicorn app_api:app --host 0.0.0.0 --port 8000
OpenWebUI 연결:
    Settings > Connections > OpenAI API
      Base URL : http://localhost:8000/v1
      API Key  : (아무 값)  -> 모델 'aml-sar' 선택 후 계좌번호(node_idx) 입력
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # neo4j_config 등이 import 시점에 env 를 읽으므로 최상단에서 먼저 호출

import json
import re
import sys
import time
from typing import Any, Generator

# Windows 콘솔/uvicorn 서브프로세스의 기본 인코딩이 cp949 면 print 의 em-dash 등
# 비-cp949 문자에서 UnicodeEncodeError 로 startup 이 죽는다 → UTF-8 로 강제.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.embedding_extractor import (
    EmbeddingExtractor,
    build_hybrid_features,
    hybrid_feature_names,
    predict_fraud_probs,
    apply_temperature,
)
from analysis.shap_analyzer import ShapAnalyzer
from reporters.ai_report_generator import build_sar_payload
from reporters.context_builder import ContextBuilder
from reporters.report_runner import ReportRunner
from reporters.sar_graph import build_sar_graph
from knowledge.rag_knowledge_base import KnowledgeBase
from knowledge.graph_rag import get_graph_retriever


# ======================================================================
# 리소스 로딩 (서버 기동 시 1회) — Streamlit 캐시 대신 모듈 전역
# ======================================================================

class _Resources:
    """모델·그래프·지식 리소스 + 조립된 파이프라인 핸들."""

    def __init__(self) -> None:
        print("[api] 모델/그래프 로딩...")
        self.graph_dict = torch.load("model/saml_graph.pt", map_location="cpu")
        in_dim = self.graph_dict["x"].shape[1]
        self.extractor = EmbeddingExtractor(in_channels=in_dim, hidden_channels=128, embed_dim=64)
        self.extractor.load_state_dict(torch.load("model/saml_gnn.pth", map_location="cpu"))
        self.extractor.eval()

        xgb_data = joblib.load("model/saml_fraud_model.pkl")
        self.xgb_model = xgb_data["xgb_model"]
        self.all_feature_names = xgb_data["all_feature_names"]
        self.scaler = xgb_data.get("scaler")
        self.temperature = float(xgb_data.get("temperature", 1.0))
        # 저장된 [emb,orig] 순서를 실제 학습 [orig,emb] 순서로 재정렬 (SHAP 라벨 정합)
        self.feature_names = hybrid_feature_names(self.all_feature_names)
        self.n_node_feats = in_dim

        print("[api] 전체 노드 임베딩 사전 계산...")
        with torch.no_grad():
            self.all_embs = self.extractor(
                self.graph_dict["x"], self.graph_dict["edge_index"]
            ).numpy()

        # 위험도 순위(테스트 노드) — watchlist 용
        test_idx = np.where(self.graph_dict["test_mask"].numpy())[0]
        probs = predict_fraud_probs(
            self.all_embs[test_idx], self.graph_dict["x"][test_idx].numpy(),
            self.xgb_model, self.scaler, self.temperature,
        )
        order = np.argsort(probs)[::-1]
        self.watchlist = [(int(test_idx[i]), float(probs[i])) for i in order]

        print("[api] 지식 리소스(RAG/GraphRAG) 준비...")
        kb, rag_ok = None, False
        try:
            kb = KnowledgeBase(pdf_directory="knowledge_base",
                               persist_dir="chroma_db", embed_model="nomic-embed-text")
            kb.build()
            rag_ok = True
        except Exception as exc:  # RAG 실패해도 탐지/그래프는 동작
            print(f"[api] RAG 초기화 경고: {exc}")
        retriever = get_graph_retriever(lazy=True)
        graph_ok = retriever.is_configured

        self.context_builder = ContextBuilder(
            knowledge_base=kb, graph_retriever=retriever,
            rag_available=rag_ok, graph_available=graph_ok,
        )
        self.report_runner = ReportRunner(self.context_builder)
        self.sar_graph = build_sar_graph(self.context_builder, self.report_runner)
        print(f"[api] 준비 완료 — 노드 {self.graph_dict['x'].shape[0]:,}, "
              f"watchlist {len(self.watchlist):,}, RAG={rag_ok}, Graph={graph_ok}")


R: _Resources | None = None  # 기동 시 채워짐


def _tier(p: float) -> str:
    return "고위험" if p >= 0.9 else "위험" if p >= 0.7 else "주의" if p >= 0.4 else "낮음"


def _analyze(node_idx: int) -> dict[str, Any]:
    """단일 계좌 탐지 — app.py 5-1~5-6 과 동일 전처리(학습 정합)."""
    assert R is not None
    n_nodes = R.graph_dict["x"].shape[0]
    if not (0 <= node_idx < n_nodes):
        raise HTTPException(404, f"node_idx 범위를 벗어남 (0 ~ {n_nodes - 1})")

    node_emb = R.all_embs[node_idx].reshape(1, -1)
    node_orig = R.graph_dict["x"][node_idx].reshape(1, -1).numpy()
    X_input = build_hybrid_features(node_orig, node_emb)
    if R.scaler is not None:
        X_input = R.scaler.transform(X_input).astype(np.float32)
    prob = float(apply_temperature(R.xgb_model.predict_proba(X_input)[:, 1], R.temperature)[0])

    analyzer = ShapAnalyzer(R.xgb_model)
    analyzer.build_explainer(X_input)
    shap_values = analyzer.compute_shap_values(X_input)

    orig_feature_names = R.feature_names[:R.n_node_feats]
    orig_df_dict = {f: float(node_orig[0][i]) for i, f in enumerate(orig_feature_names)}
    sar_payload, sar_json_str = build_sar_payload(
        selected_idx=node_idx, prob=prob, shap_values=shap_values,
        X_input=X_input, all_feature_names=R.feature_names, orig_df_dict=orig_df_dict,
    )

    sv = shap_values[0]
    top = np.argsort(np.abs(sv))[::-1][:5]
    factors = [{"feature": R.feature_names[i], "shap_value": round(float(sv[i]), 4),
                "direction": "위험 증가" if sv[i] > 0 else "위험 감소"} for i in top]

    return {
        "node_idx": node_idx,
        "fraud_probability": round(prob, 4),
        "fraud_probability_display": f"{min(prob, 0.999) * 100:.1f}%",
        "risk_tier": _tier(prob),
        "top_factors": factors,
        "_sar_payload": sar_payload,
        "_sar_json_str": sar_json_str,
    }


def _generate_sar(node_idx: int) -> dict[str, Any]:
    """LangGraph 파이프라인 1회 실행 -> 완성 SAR + 메타."""
    assert R is not None
    a = _analyze(node_idx)
    inputs = {"sar_payload": a["_sar_payload"], "sar_json_str": a["_sar_json_str"],
              "node_idx": node_idx}
    final_state: dict[str, Any] = dict(inputs)
    node_flow: list[str] = []
    for event in R.sar_graph.stream(inputs, stream_mode="updates"):
        name = next(iter(event))
        if name == "__end__":
            continue
        node_flow.append(name)
        final_state.update(event[name])
    return {
        "node_idx": node_idx,
        "fraud_probability_display": a["fraud_probability_display"],
        "risk_tier": a["risk_tier"],
        "query_type": final_state.get("query_type", ""),
        "is_valid": final_state.get("is_valid"),
        "node_flow": node_flow,
        "sar_report": final_state.get("final_report", ""),
    }


# ======================================================================
# FastAPI 앱
# ======================================================================

app = FastAPI(title="AML 탐지 / SAR 생성 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global R
    if R is None:
        R = _Resources()


# ── REST ──────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": R is not None,
            "nodes": int(R.graph_dict["x"].shape[0]) if R else 0}


@app.get("/api/accounts/watchlist")
def watchlist(limit: int = 20) -> dict:
    assert R is not None
    items = [{"node_idx": idx, "fraud_probability": round(p, 4),
              "fraud_probability_display": f"{min(p, 0.999) * 100:.1f}%",
              "risk_tier": _tier(p)} for idx, p in R.watchlist[:max(1, min(limit, 200))]]
    return {"count": len(items), "accounts": items}


@app.get("/api/accounts/{node_idx}")
def account_detail(node_idx: int) -> dict:
    a = _analyze(node_idx)
    return {k: v for k, v in a.items() if not k.startswith("_")}


class SarRequest(BaseModel):
    node_idx: int


@app.post("/api/sar/generate")
def sar_generate(req: SarRequest) -> dict:
    return _generate_sar(req.node_idx)


# ── OpenAI 호환 (OpenWebUI) ────────────────────────────────────────────

_MODEL_ID = "aml-sar"


@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [
        {"id": _MODEL_ID, "object": "model", "created": int(time.time()),
         "owned_by": "aml-project"}]}


class ChatMessage(BaseModel):
    role: str
    content: Any  # str 또는 멀티모달 파트 리스트


class ChatRequest(BaseModel):
    model: str | None = _MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # 멀티모달 파트 -> text 만 추출
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _help_text() -> str:
    assert R is not None
    top = R.watchlist[:8]
    lines = "\n".join(f"- `{idx}`  (의심도 {min(p, 0.999) * 100:.1f}%, {_tier(p)})"
                      for idx, p in top)
    return ("계좌 번호(node_idx)를 입력하면 해당 계좌의 자금세탁 의심거래보고서(SAR)를 생성합니다.\n\n"
            "예시로 분석해 볼 만한 고위험 계좌:\n" + lines +
            "\n\n번호만 입력하거나 '계좌 5432 분석' 처럼 입력하세요.")


def _extract_node_idx(text: str) -> int | None:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _openai_chunk(content: str | None, finish: str | None = None) -> str:
    delta = {} if content is None else {"content": content}
    payload = {"id": "chatcmpl-aml", "object": "chat.completion.chunk",
               "created": int(time.time()), "model": _MODEL_ID,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sar_text(node_idx: int) -> str:
    """탐지 헤더 + 완성 SAR (OpenWebUI 표시용 마크다운)."""
    result = _generate_sar(node_idx)
    header = (f"## 탐지 결과 — 계좌 {node_idx}\n"
              f"- 자금세탁 의심도: **{result['fraud_probability_display']}** "
              f"({result['risk_tier']})\n"
              f"- 질의 유형: {result['query_type']} · "
              f"법령 근거 포함: {result['is_valid']}\n"
              f"- 파이프라인: {' → '.join(result['node_flow'])}\n\n---\n\n")
    return header + result["sar_report"]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    text = _last_user_text(req.messages)
    node_idx = _extract_node_idx(text)

    if node_idx is None:
        body = _help_text()
    else:
        try:
            body = _sar_text(node_idx)
        except HTTPException as exc:
            body = f"오류: {exc.detail}"
        except Exception as exc:  # LLM/네트워크 오류 -> 사용자에게 메시지로 전달
            body = f"SAR 생성 중 오류가 발생했습니다: {exc}"

    if not req.stream:
        return {"id": "chatcmpl-aml", "object": "chat.completion",
                "created": int(time.time()), "model": _MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": body}}]}

    def gen() -> Generator[str, None, None]:
        yield _openai_chunk("")  # role 시작
        # 줄 단위로 흘려보내 스트리밍 UX 부여
        for line in body.splitlines(keepends=True):
            yield _openai_chunk(line)
        yield _openai_chunk(None, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
