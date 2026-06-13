import json
import os
import time
from datetime import datetime
from typing import Any, Generator

import numpy as np

from reporters.sar_template import SECTION_FORMAT_INSTRUCTIONS


# ======================================================================
# Groq 클라이언트 팩토리
# ======================================================================

def _get_groq_client():
    """
    Groq 클라이언트 인스턴스를 반환합니다.

    우선순위:
    1. Streamlit secrets  — st.secrets["groq"]["api_key"]  (클라우드 배포)
    2. 환경변수           — GROQ_API_KEY                   (Docker / CI)

    Raises:
        EnvironmentError: API 키를 찾을 수 없을 때
        ImportError     : groq 패키지 미설치 시
    """
    try:
        from groq import Groq  # type: ignore
    except ImportError as e:
        raise ImportError(
            "groq 패키지가 설치되지 않았습니다. `pip install groq` 를 실행하세요."
        ) from e

    api_key: str | None = None

    # 1) Streamlit secrets
    try:
        import streamlit as st  # type: ignore
        api_key = st.secrets.get("groq", {}).get("api_key")
    except Exception:
        pass

    # 2) 환경변수
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY가 설정되지 않았습니다.\n"
            "• 로컬: 환경변수 GROQ_API_KEY 를 설정하세요.\n"
            "• Streamlit Cloud: App settings > Secrets 에 [groq] api_key 를 추가하세요.\n"
            "• 무료 키 발급: https://console.groq.com"
        )

    return Groq(api_key=api_key)


# ======================================================================
# 내부 유틸: 프롬프트 조립
# ======================================================================

# Groq(chat completions)는 응답에 프롬프트를 에코하지 않으므로
# 마커는 _strip_rag_from_output()의 ##II## 탐지 로직이 대신 처리합니다.
_RAG_OUTPUT_MARKER = "=== 섹션 생성 시작 ==="


def _build_messages(
    json_data: str,
    rag_context: str = "",
    graph_context: str = "",
) -> list[dict]:
    """
    Groq Chat Completions API용 messages 리스트를 조립합니다.

    system 메시지: AML 수사관 역할 정의 + 섹션 출력 형식 지시
    user   메시지: RAG·GraphRAG 참조 자료 + 분석 데이터 JSON

    이 구조는 chat 모델의 instruction-following 능력을 최대한 활용하며,
    _strip_rag_from_output()의 ##II## 탐지로 노이즈를 제거합니다.
    """
    # ── system: 역할 + 형식 지시 ─────────────────────────────────────
    system_content = (
        "당신은 대한민국 금융정보분석원(KoFIU) 소속의 자금세탁방지(AML) 전문 수사관입니다.\n"
        "아래 지시에 따라 SAR 보고서의 각 섹션 내용을 작성하십시오.\n"
        "\n"
        + SECTION_FORMAT_INSTRUCTIONS
    )

    # ── user: 참조 자료 + 분석 데이터 ───────────────────────────────
    # Groq 무료 티어 TPM 제한(6,000) 대비 컨텍스트 길이 제한
    # 한국어 1자 ≈ 1.5~2 토큰 기준으로 보수적으로 잡음
    _MAX_RAG_CHARS   = 800   # ≈ 400~500 토큰
    _MAX_GRAPH_CHARS = 400   # ≈ 200~250 토큰

    user_parts: list[str] = []

    if rag_context.strip():
        truncated_rag = rag_context[:_MAX_RAG_CHARS]
        if len(rag_context) > _MAX_RAG_CHARS:
            truncated_rag += "\n...(이하 생략)"
        user_parts.append(
            "【법령·지침 참조 자료 (KoFIU 공식 문서)】\n"
            "아래 법령·지침을 보고서 작성의 근거로 활용하십시오.\n\n"
            + truncated_rag
        )

    if graph_context.strip():
        truncated_graph = graph_context[:_MAX_GRAPH_CHARS]
        if len(graph_context) > _MAX_GRAPH_CHARS:
            truncated_graph += "\n...(이하 생략)"
        user_parts.append(
            "【거래 네트워크 구조 분석 결과 (Neo4j GraphRAG)】\n"
            "아래는 해당 계좌의 실제 거래 그래프에서 도출된 네트워크 구조 정보입니다.\n"
            "II 섹션(의심 징후)과 III 섹션(위험 평가)에 구체적인 수치와 경로로 반영하십시오.\n\n"
            + truncated_graph
        )

    user_parts.append("[분석 데이터]\n" + json_data)
    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


# ======================================================================
# 내부 유틸: Groq 호출 재시도 (무료 티어 TPM 초과 대응)
# ======================================================================

_MAX_RETRIES = 2     # 최초 1회 + 재시도 2회
_RETRY_CAP   = 30.0  # 단일 대기 상한 (초) — 데모 무한 대기 방지


def _is_rate_limit_error(exc: Exception) -> bool:
    """예외가 Groq TPM/RPM 초과(429)인지 판별한다.

    groq.RateLimitError 를 직접 import 하지 않고 클래스명과 status_code 로
    판별한다 (SDK 버전 간 예외 경로 차이에 견고하도록).
    """
    if exc.__class__.__name__ == "RateLimitError":
        return True
    return getattr(exc, "status_code", None) == 429


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """429 응답의 Retry-After 헤더를 우선 사용하고, 없으면 default 를 쓴다."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return default


def _completion_with_retry(client, **kwargs):
    """client.chat.completions.create 를 TPM 초과 시 지수 백오프로 재시도한다.

    무료 티어에서 RAG 컨텍스트가 길면 429(RateLimitError)가 발생한다.
    Retry-After 헤더(있으면) 또는 5s·10s 백오프로 제한 횟수만 재시도하고,
    그 외 예외이거나 재시도 소진 시 그대로 전파한다.
    스트리밍 경로에서도 스트림 생성 시점의 429 를 잡아 안정성을 높인다.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == _MAX_RETRIES:
                raise
            wait = min(_retry_after_seconds(exc, default=5.0 * (attempt + 1)), _RETRY_CAP)
            time.sleep(wait)


def _strip_rag_from_output(text: str) -> str:
    """
    LLM이 RAG 입력 섹션이나 마커를 출력에 포함한 경우 제거합니다.

    처리 순서:
    1. '=== 섹션 생성 시작 ===' 마커 이후만 추출 (Ollama 호환 잔여 처리)
    2. '---[내부 참조 자료...]---' 블록 제거
    3. 첫 번째 섹션 마커(##II##) 이전의 잡음을 항상 제거
       (임계값 없이 마커 직전까지 모두 제거 → 일관된 시작점 보장)
    """
    import re as _re

    # 1) 스트리밍 시작 마커 이후만 추출
    if _RAG_OUTPUT_MARKER in text:
        text = text.split(_RAG_OUTPUT_MARKER, 1)[-1].strip()

    # 2) 내부 참조 자료 블록 제거 (변형 패턴 포함)
    noise_patterns = [
        r"---\[내부 참조 자료[^\]]*\]---",
        r"\[내부 참조 자료[^\]]*\]",
        r"---\[분석 데이터\]---",
    ]
    for pat in noise_patterns:
        text = _re.sub(pat, "", text, flags=_re.DOTALL)

    # 3) ##II## 마커 이전의 모든 잡음 제거 (항상 적용)
    first_marker_idx = text.find("##II##")
    if first_marker_idx > 0:
        text = text[first_marker_idx:]

    return text.strip()


# ======================================================================
# 피처 값 가독성 변환 유틸 (변경 없음)
# ======================================================================

def _humanize_feature_value(z_score: float, feature_name: str) -> dict:
    """
    StandardScaler로 정규화된 z-score 값을 사람이 이해하기 쉬운
    정성적 수준(level)과 설명(description)으로 변환합니다.
    """
    z = float(z_score)

    if z >= 2.0:
        level = "매우 높음"
    elif z >= 1.0:
        level = "높음"
    elif z >= 0.0:
        level = "평균 이상"
    elif z >= -1.0:
        level = "평균 이하"
    elif z >= -2.0:
        level = "낮음"
    else:
        level = "매우 낮음"

    if feature_name.startswith("GNN_Emb"):
        if z >= 1.5:
            desc = "강한 사기 방향 활성화 (네트워크상 의심 패턴 밀집 영역)"
        elif z >= 0.5:
            desc = "중간 사기 방향 활성화 (주변 의심 계좌와 유사한 거래 패턴)"
        elif z >= -0.5:
            desc = "중립적 활성화 (정상·사기 경계 영역)"
        elif z >= -1.5:
            desc = "중간 정상 방향 활성화 (주변 정상 계좌와 유사한 거래 패턴)"
        else:
            desc = "강한 정상 방향 활성화 (네트워크상 정상 패턴 밀집 영역)"
        return {"수준": level, "z점수": round(z, 4), "설명": desc}

    feature_context: dict[str, tuple[str, str]] = {
        "amount":           ("대규모 거래 금액",              "소규모 거래 금액"),
        "log_amount":       ("대규모 거래 금액 (로그)",        "소규모 거래 금액 (로그)"),
        "balance_mismatch": ("잔액 불일치 심각 (자금 은닉 의심)", "잔액 정상 범위"),
        "hour_of_day_norm": ("비정상 시간대 거래 (심야·새벽)", "일반 업무 시간대 거래"),
        "step":             ("장기 거래 기록",                "초기 또는 단기 거래 기록"),
        "oldbalanceOrg":    ("송금 전 잔액 높음",             "송금 전 잔액 낮음"),
        "newbalanceOrig":   ("송금 후 잔액 높음",             "송금 후 잔액 낮음 (계좌 비우기 의심)"),
        "oldbalanceDest":   ("수신 전 잔액 높음",             "수신 전 잔액 낮음"),
        "newbalanceDest":   ("수신 후 잔액 높음",             "수신 후 잔액 낮음"),
        "isFraud":          ("사기 거래 비율 높음",           "정상 거래 비율 높음"),
    }

    ctx_high, ctx_low = feature_context.get(
        feature_name,
        (f"{feature_name} 수치 높음", f"{feature_name} 수치 낮음"),
    )
    desc = ctx_high if z >= 0 else ctx_low

    if abs(z) >= 2.0:
        desc = "[주의] " + desc + " (극단적 이상치)"
    elif abs(z) >= 1.0:
        desc = desc + " (주목할 수준)"

    return {"수준": level, "z점수": round(z, 4), "설명": desc}


# ======================================================================
# 1-A. 스트리밍 API (권장 — Streamlit write_stream 연동)
# ======================================================================

def stream_ai_report(
    json_data: str,
    model: str = "llama-3.1-8b-instant",
    rag_context: str = "",
    graph_context: str = "",
    # 하위 호환 인자 (무시됨 — Groq SDK가 연결을 관리)
    ollama_url: str | None = None,
    connect_timeout: int = 10,
) -> Generator[str, None, None]:
    """
    Groq Streaming API로 SAR 보고서를 토큰 단위로 생성합니다.

    Groq LPU는 llama3.1(8B) 기준 초당 ~700 토큰을 처리하며,
    스트리밍으로 Streamlit st.write_stream()과 즉시 연동됩니다.

    Args:
        json_data     (str): SAR JSON 문자열
        model         (str): Groq 모델명 (기본: llama-3.1-8b-instant)
                             고품질 옵션: "llama-3.3-70b-versatile"
        rag_context   (str): KoFIU 법령·지침 텍스트 RAG 컨텍스트
        graph_context (str): Neo4j 거래 네트워크 GraphRAG 컨텍스트
        ollama_url    (str): 사용되지 않음 (하위 호환용)
        connect_timeout (int): 사용되지 않음 (하위 호환용)

    Yields:
        str: 토큰 단위 텍스트 조각

    Raises:
        EnvironmentError: GROQ_API_KEY 미설정
        groq.APIError   : Groq API 오류
    """
    client   = _get_groq_client()
    messages = _build_messages(json_data, rag_context, graph_context)

    stream = _completion_with_retry(
        client,
        model=model,
        messages=messages,
        stream=True,
        temperature=0.1,
        seed=42,
        max_tokens=2048,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token


# ======================================================================
# 1-B. 비스트리밍 API (폴백용)
# ======================================================================

def generate_ai_report(
    json_data: str,
    model: str = "llama-3.1-8b-instant",
    rag_context: str = "",
    graph_context: str = "",
    # 하위 호환 인자 (무시됨)
    ollama_url: str | None = None,
    timeout: int = 300,
) -> str:
    """
    Groq API를 단일 요청으로 호출해 SAR 보고서 전체를 반환합니다.

    Streamlit에서는 stream_ai_report() 사용을 권장합니다.

    Args:
        json_data     (str): SAR JSON 문자열
        model         (str): Groq 모델명 (기본: llama-3.1-8b-instant)
        rag_context   (str): KoFIU 법령·지침 텍스트 RAG 컨텍스트
        graph_context (str): Neo4j 거래 네트워크 GraphRAG 컨텍스트
        ollama_url    (str): 사용되지 않음 (하위 호환용)
        timeout       (int): 사용되지 않음 (하위 호환용)

    Returns:
        str: 생성된 보고서 텍스트 (실패 시 오류 메시지)
    """
    try:
        client   = _get_groq_client()
        messages = _build_messages(json_data, rag_context, graph_context)

        response = _completion_with_retry(
            client,
            model=model,
            messages=messages,
            stream=False,
            temperature=0.1,
            seed=42,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or "보고서 내용 생성 실패"
        raw = _strip_rag_from_output(raw)
        raw = raw.replace("があります", "가 있습니다")
        raw = raw.replace("必要があります", "필요가 있습니다")
        return raw

    except EnvironmentError as exc:
        return f"[오류] API 키 오류: {exc}"
    except Exception as exc:
        # groq.APIConnectionError, groq.RateLimitError 등
        err_type = type(exc).__name__
        return f"[오류] Groq API 오류 ({err_type}): {exc}"


# ======================================================================
# 2. SAR 페이로드 조립 (변경 없음)
# ======================================================================

def build_sar_payload(
    selected_idx: int,
    prob: float,
    shap_values: np.ndarray,
    X_input: np.ndarray,
    all_feature_names: list[str],
    orig_df_dict: dict[str, Any],
) -> tuple[dict, str]:
    """
    SHAP 분석 결과와 예측 확률을 바탕으로 Groq 전송용 SAR 데이터셋을 구성합니다.

    Args:
        selected_idx      (int)    : 분석 대상 노드 인덱스
        prob              (float)  : XGBoost 예측 사기 확률
        shap_values       (ndarray): SHAP 값 배열 [1, n_features]
        X_input           (ndarray): 모델 입력 배열 [1, n_features]
        all_feature_names (list)   : 전체 피처명 리스트
        orig_df_dict      (dict)   : 원본 피처 10개의 {feature: value} 딕셔너리

    Returns:
        tuple[dict, str]: (sar_payload, sar_json_str)
    """
    top_indices = np.argsort(np.abs(shap_values[0]))[-5:][::-1]
    top_features_summary = []

    for idx in top_indices:
        feature_name = all_feature_names[idx]
        feature_val  = float(X_input[0, idx])
        shap_val     = float(shap_values[0][idx])

        f_type    = "네트워크 관계 특징" if feature_name.startswith("GNN_Emb") else "개별 거래 특징"
        direction = "위험 증가(사기 의심)" if shap_val > 0 else "위험 감소(정상 의심)"
        human     = _humanize_feature_value(feature_val, feature_name)

        top_features_summary.append({
            "특징명":   feature_name,
            "특징유형": f_type,
            "현재값":   f"{human['z점수']:+.4f}",
            "수준":     f"{human['수준']}  (z={human['z점수']:+.4f})",
            "값 설명":  human["설명"],
            "영향도":   direction,
        })

    if prob > 0.7:
        risk_level_str = "고위험(High)"
    elif prob > 0.4:
        risk_level_str = "중위험(Medium)"
    else:
        risk_level_str = "저위험(Low)"

    sar_payload = {
        "report_context": {
            "analysis_date":    datetime.now().strftime("%Y-%m-%d"),
            "target_node_id":   int(selected_idx),
            "fraud_probability": f"{prob:.2%}",
            "risk_level":       risk_level_str,
        },
        "key_risk_factors": top_features_summary,
        "raw_feature_data": orig_df_dict,
        "model_explanation": (
            "이 모델은 거래처 간의 송금 네트워크(GNN)와 "
            "해당 계좌의 통계적 특징(XGBoost)을 결합하여 분석한 결과입니다."
        ),
    }

    sar_json_str = json.dumps(sar_payload, indent=2, ensure_ascii=False)
    return sar_payload, sar_json_str
