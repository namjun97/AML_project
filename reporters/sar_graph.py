from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import StateGraph, END

from reporters.context_builder import ContextBuilder

if TYPE_CHECKING:
    from reporters.report_runner import ReportRunner


def _log_validation_feedback(is_valid: bool, query_type: str) -> None:
    """validate 결과를 LangSmith 피드백으로 기록합니다 (best-effort).

    LANGCHAIN_TRACING_V2 가 꺼져 있거나 트레이스 컨텍스트 밖이면 조용히 스킵.
    모니터링 실패가 SAR 생성 자체를 막아서는 안 되므로 모든 예외를 삼킵니다.
    """
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        return
    try:
        from langsmith import Client
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is None:
            return

        Client().create_feedback(
            run_id=run_tree.trace_id,   # 루트 트레이스에 피드백 부착
            key="law_reference_check",
            score=1.0 if is_valid else 0.0,
            comment=f"법령 키워드 검증 (query_type={query_type})",
        )
    except Exception:
        pass


# ======================================================================
# State
# ======================================================================

class AMLState(TypedDict):
    sar_payload:         dict        # build_sar_payload() 결과
    sar_json_str:        str         # JSON 직렬화 문자열
    node_idx:            int         # 분석 대상 노드 인덱스
    query_type:          str         # "network" | "standard"
    rag_scores:          list        # RAG 검색 원본 결과 [{text, source, score}, ...]
    rag_context:         str         # integrate 가 점수 내림차순 정렬 후 포맷 (LLM 프롬프트용)
    graph_context:       str         # Graph RAG 포맷 텍스트
    sar_draft:           str         # LLM 생성 SAR 초안
    final_report:        str         # 검증·조립 완성 보고서
    is_valid:            bool        # 법령 근거 포함 여부


# ======================================================================
# 그래프 팩토리
# ======================================================================

def build_sar_graph(
    context_builder: "ContextBuilder",
    report_runner:   "ReportRunner",
):
    """
    SAR 생성 LangGraph 워크플로우를 빌드합니다.

    기존 ContextBuilder / ReportRunner 인스턴스를 주입받아
    각 노드 함수가 클로저로 참조합니다.

    Returns:
        CompiledGraph: workflow.compile() 결과. invoke() 또는 stream()으로 실행.

    Example:
        graph = build_sar_graph(context_builder, report_runner)
        result = graph.invoke({
            "sar_payload":  sar_payload,
            "sar_json_str": sar_json_str,
            "node_idx":     node_idx,
        })
        print(result["final_report"])
        print("법령 근거 포함:", result["is_valid"])
        print("평균 RAG 유사도:", sum(r["score"] for r in result["rag_scores"]) / len(result["rag_scores"]))
    """

    # ------------------------------------------------------------------
    # Node 1: 질의 분류
    # ------------------------------------------------------------------
    def classify_query(state: AMLState) -> dict:
        has_network = ContextBuilder.is_network_query(state["sar_payload"])
        return {"query_type": "network" if has_network else "standard"}

    # ------------------------------------------------------------------
    # Node 2a: Hybrid 검색 — "network" 경로 (Vector RAG + Graph RAG)
    #   GNN 임베딩이 위험 신호인 경우 거래 네트워크 구조가 핵심 근거이므로
    #   Neo4j Graph RAG 를 함께 조회한다.
    #   포맷은 하지 않고 원본 점수 결과(rag_scores)만 넘긴다 — 컨텍스트
    #   조립은 integrate 노드가 단독으로 담당한다 (단일 책임).
    # ------------------------------------------------------------------
    def retrieve_hybrid(state: AMLState) -> dict:
        return {
            "rag_scores":    context_builder.search_rag_scored(state["sar_payload"]),
            "graph_context": context_builder.build_graph_context(state["node_idx"]),
        }

    # ------------------------------------------------------------------
    # Node 2b: Vector 전용 검색 — "standard" 경로
    #   네트워크 패턴이 위험 신호가 아닌 경우 Neo4j 조회를 생략한다.
    #   (AuraDB TLS 초기 연결이 최대 30초 — 불필요한 지연 제거)
    # ------------------------------------------------------------------
    def retrieve_vector(state: AMLState) -> dict:
        return {
            "rag_scores":    context_builder.search_rag_scored(state["sar_payload"]),
            "graph_context": "",
        }

    # ------------------------------------------------------------------
    # Node 3: 컨텍스트 조립 (점수 가중 앙상블 — 단일 책임)
    #   rag_scores 를 유사도 내림차순 정렬한 뒤 format_scored_context 로
    #   LLM 프롬프트용 rag_context 를 생성한다. 이 정렬이 실질적 의미를
    #   갖는 이유: ai_report_generator._build_messages 가 RAG 컨텍스트를
    #   상위 800자로 truncate 하므로, 고유사도 법령 구절이 먼저 와야
    #   잘림에서 살아남아 실제로 LLM 에 전달된다.
    #   (retrieve → 원본 점수 결과 / integrate → 정렬·포맷 / generate → 소비)
    # ------------------------------------------------------------------
    def integrate_context(state: AMLState) -> dict:
        rag_scores   = state.get("rag_scores") or []
        sorted_scores = sorted(
            rag_scores, key=lambda r: r.get("score", 0), reverse=True
        )
        return {"rag_context": context_builder.format_scored_context(sorted_scores)}

    # ------------------------------------------------------------------
    # Node 4: SAR 초안 생성 (Groq LLM)
    # ------------------------------------------------------------------
    def generate_sar(state: AMLState) -> dict:
        raw_text = ""
        for token in report_runner.stream(
            state["sar_json_str"],
            state["rag_context"],
            state["graph_context"],
        ):
            raw_text += token
        return {"sar_draft": raw_text}

    # ------------------------------------------------------------------
    # Node 5: 검증 + 최종 조립
    # ------------------------------------------------------------------
    _LAW_KEYWORDS = [
        "특정금융정보법", "자금세탁방지", "의심거래보고",
        "금융정보분석원", "KoFIU", "STR", "특금법",
    ]

    def validate_output(state: AMLState) -> dict:
        draft        = state.get("sar_draft", "")
        is_valid     = any(kw in draft for kw in _LAW_KEYWORDS)
        final_report = report_runner.finalize(
            draft,
            state["sar_payload"],
            state["graph_context"],
        )
        _log_validation_feedback(is_valid, state.get("query_type", ""))
        return {"is_valid": is_valid, "final_report": final_report}

    # ------------------------------------------------------------------
    # 그래프 조립
    # ------------------------------------------------------------------
    workflow = StateGraph(AMLState)

    workflow.add_node("classify",        classify_query)
    workflow.add_node("retrieve",        retrieve_hybrid)   # network 경로
    workflow.add_node("retrieve_vector", retrieve_vector)   # standard 경로
    workflow.add_node("integrate",       integrate_context)
    workflow.add_node("generate",        generate_sar)
    workflow.add_node("validate",        validate_output)

    workflow.set_entry_point("classify")

    # 조건부 분기: 질의 유형에 따라 검색 전략 선택
    #   "network"  → Hybrid (Vector + Neo4j Graph RAG)
    #   "standard" → Vector 전용 (Neo4j 호출 생략)
    workflow.add_conditional_edges(
        "classify",
        lambda state: state["query_type"],
        {
            "network":  "retrieve",
            "standard": "retrieve_vector",
        },
    )

    workflow.add_edge("retrieve",        "integrate")
    workflow.add_edge("retrieve_vector", "integrate")
    workflow.add_edge("integrate",       "generate")
    workflow.add_edge("generate",        "validate")
    workflow.add_edge("validate",        END)

    return workflow.compile()
