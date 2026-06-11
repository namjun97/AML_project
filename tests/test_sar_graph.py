"""
Unit tests for reporters/sar_graph.py — LangGraph SAR pipeline.

Each test isolates one node by mocking ContextBuilder and ReportRunner,
then invoking the compiled graph with controlled state input.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_mocks(
    rag_scores=None,
    rag_context="법률 근거: 특정금융정보법 제4조",
    graph_context="[거래 네트워크] 연결 노드 3개",
    stream_tokens=None,
    finalize_return="[최종 SAR 보고서] 특정금융정보법 기반",
):
    """Return (context_builder, report_runner) mocks with sensible defaults."""
    if rag_scores is None:
        rag_scores = [
            {"text": "특정금융정보법 제4조 의심거래보고 의무", "source": "law_01.pdf", "score": 0.85},
            {"text": "자금세탁방지 가이드라인 3.2항",          "source": "guide_02.pdf", "score": 0.72},
        ]
    if stream_tokens is None:
        stream_tokens = ["SAR 보고서: ", "특정금융정보법 ", "의심거래 확인됨."]

    cb = MagicMock()
    cb.search_rag_scored.return_value      = rag_scores
    cb.format_scored_context.return_value  = rag_context
    cb.build_graph_context.return_value    = graph_context

    rr = MagicMock()
    # side_effect 로 매 호출마다 새 이터레이터를 생성 — invoke() 재사용 시에도
    # 이터레이터가 소진되지 않아 두 번째 실행에서도 토큰이 정상 수집된다.
    rr.stream.side_effect    = lambda *a, **kw: iter(stream_tokens)
    rr.finalize.return_value = finalize_return

    return cb, rr


def _base_input(extra: dict | None = None) -> dict:
    """Minimal valid LangGraph input state."""
    state = {
        "sar_payload": {
            "report_context": {"risk_level": "고위험"},
            "key_risk_factors": [
                {"특징명": "transaction_amount", "기여도": 0.6},
                {"특징명": "GNN_Emb_0",           "기여도": 0.4},
            ],
        },
        "sar_json_str": '{"risk": "high"}',
        "node_idx": 42,
    }
    if extra:
        state.update(extra)
    return state


def _build(cb=None, rr=None):
    from reporters.sar_graph import build_sar_graph
    _cb, _rr = _make_mocks()
    return build_sar_graph(cb or _cb, rr or _rr)


# ---------------------------------------------------------------------------
# 1. classify_query node
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def test_network_when_gnn_feature_present(self):
        """GNN 임베딩 피처가 있으면 query_type == 'network'."""
        graph = _build()
        result = graph.invoke(_base_input())
        assert result["query_type"] == "network"

    def test_standard_when_no_gnn_feature(self):
        """GNN 피처가 없으면 query_type == 'standard'."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        state  = _base_input()
        state["sar_payload"]["key_risk_factors"] = [
            {"특징명": "transaction_count", "기여도": 0.9},
            {"특징명": "avg_amount",        "기여도": 0.5},
        ]
        result = graph.invoke(state)
        assert result["query_type"] == "standard"

    def test_standard_when_factors_empty(self):
        """key_risk_factors 가 빈 리스트이면 'standard'."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        state  = _base_input()
        state["sar_payload"]["key_risk_factors"] = []
        result = graph.invoke(state)
        assert result["query_type"] == "standard"


# ---------------------------------------------------------------------------
# 2. retrieve_rag node
# ---------------------------------------------------------------------------

class TestRetrieveRagNode:
    def test_calls_search_rag_scored(self):
        """retrieve_rag 가 search_rag_scored 를 sar_payload 로 호출한다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        graph.invoke(_base_input())
        cb.search_rag_scored.assert_called_once()

    def test_rag_scores_propagated(self):
        """rag_scores 가 최종 state 에 그대로 전달된다."""
        scores = [{"text": "test law", "source": "src.pdf", "score": 0.9}]
        cb, rr = _make_mocks(rag_scores=scores)
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["rag_scores"] == scores

    def test_rag_context_formatted_from_scores(self):
        """rag_context 는 format_scored_context(rag_scores) 결과로 채워진다.

        동일 쿼리로 ChromaDB 를 두 번 검색하지 않는다 (중복 검색 방지).
        """
        scores = [{"text": "법령 본문", "source": "law.pdf", "score": 0.8}]
        cb, rr = _make_mocks(rag_scores=scores, rag_context="법률 A 내용")
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["rag_context"] == "법률 A 내용"
        cb.format_scored_context.assert_called_once_with(scores)
        cb.build_rag_context.assert_not_called()

    def test_empty_rag_scores_graceful(self):
        """RAG 비가용 시 (빈 리스트) 그래프가 예외 없이 완료된다."""
        cb, rr = _make_mocks(rag_scores=[])
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["rag_scores"] == []
        assert result["final_report"] != ""


# ---------------------------------------------------------------------------
# 2a-1. 조건부 라우팅 (classify → retrieve / retrieve_vector)
# ---------------------------------------------------------------------------

class TestConditionalRouting:
    def test_network_route_calls_graph_rag(self):
        """'network' 질의는 Hybrid 경로 — Neo4j Graph RAG 를 호출한다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())  # GNN 피처 포함 → network
        cb.build_graph_context.assert_called_once()
        assert result["graph_context"] != ""

    def test_standard_route_skips_neo4j(self):
        """'standard' 질의는 Vector 전용 경로 — Neo4j 호출을 생략한다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        state  = _base_input()
        state["sar_payload"]["key_risk_factors"] = [
            {"특징명": "transaction_count", "기여도": 0.9},
        ]
        result = graph.invoke(state)
        cb.build_graph_context.assert_not_called()
        assert result["graph_context"] == ""

    def test_standard_route_still_completes_pipeline(self):
        """standard 경로도 generate → validate 까지 정상 완료된다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        state  = _base_input()
        state["sar_payload"]["key_risk_factors"] = []
        result = graph.invoke(state)
        assert result["final_report"] != ""
        assert isinstance(result["is_valid"], bool)
        assert result["rag_scores"]  # Vector RAG 는 양쪽 경로 모두 실행


# ---------------------------------------------------------------------------
# 2b. ContextBuilder.format_scored_context (실제 구현 검증 — mock 아님)
# ---------------------------------------------------------------------------

class TestFormatScoredContext:
    def test_threshold_filters_low_scores(self):
        """score_threshold 미만 결과는 출력에서 제외된다."""
        from reporters.context_builder import ContextBuilder
        results = [
            {"text": "고점수 법령", "source": "high.pdf", "score": 0.80},
            {"text": "저점수 법령", "source": "low.pdf",  "score": 0.10},
        ]
        out = ContextBuilder.format_scored_context(results, score_threshold=0.3)
        assert "고점수 법령" in out
        assert "저점수 법령" not in out

    def test_output_format_matches_knowledge_base(self):
        """KnowledgeBase.format_context() 와 동일한 '[참고 N]' 형식을 유지한다."""
        from reporters.context_builder import ContextBuilder
        results = [{"text": "본문", "source": "law.pdf", "score": 0.85}]
        out = ContextBuilder.format_scored_context(results)
        assert out == "[참고 1] 출처: law.pdf (유사도: 0.85)\n본문"

    def test_empty_results_return_empty_string(self):
        """검색 결과가 없으면 빈 문자열을 반환한다."""
        from reporters.context_builder import ContextBuilder
        assert ContextBuilder.format_scored_context([]) == ""


# ---------------------------------------------------------------------------
# 3. integrate_context node
# ---------------------------------------------------------------------------

class TestIntegrateContextNode:
    def test_integrated_context_contains_both_sources(self):
        """integrated_context 에 RAG 와 Graph RAG 내용이 모두 포함된다."""
        cb, rr = _make_mocks(
            rag_context="RAG 법령 내용",
            graph_context="Graph 네트워크 데이터",
        )
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        ic = result["integrated_context"]
        assert "RAG" in ic
        assert "Graph" in ic

    def test_scores_sorted_descending_in_integrated_context(self):
        """유사도 낮은 항목이 높은 항목보다 integrated_context 에서 뒤에 온다."""
        scores = [
            {"text": "낮은 점수 법령",   "source": "low.pdf",  "score": 0.50},
            {"text": "높은 점수 법령",   "source": "high.pdf", "score": 0.90},
        ]
        cb, rr = _make_mocks(rag_scores=scores)
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        ic = result["integrated_context"]
        assert ic.index("높은 점수 법령") < ic.index("낮은 점수 법령")

    def test_empty_graph_context_omitted(self):
        """graph_context 가 없으면 integrated_context 에서 [Graph RAG] 섹션이 없다."""
        cb, rr = _make_mocks(graph_context="")
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert "Graph RAG" not in result["integrated_context"]


# ---------------------------------------------------------------------------
# 4. generate_sar node
# ---------------------------------------------------------------------------

class TestGenerateSarNode:
    def test_stream_tokens_concatenated(self):
        """report_runner.stream() 토큰들이 sar_draft 로 이어붙여진다."""
        cb, rr = _make_mocks(stream_tokens=["토큰A ", "토큰B ", "토큰C"])
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["sar_draft"] == "토큰A 토큰B 토큰C"

    def test_stream_called_with_sar_json_str(self):
        """report_runner.stream 첫 인자가 sar_json_str 이다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        state  = _base_input()
        graph.invoke(state)
        call_args = rr.stream.call_args[0]
        assert call_args[0] == state["sar_json_str"]


# ---------------------------------------------------------------------------
# 5. validate_output node
# ---------------------------------------------------------------------------

class TestValidateOutputNode:
    @pytest.mark.parametrize("keyword", [
        "특정금융정보법", "자금세탁방지", "의심거래보고",
        "금융정보분석원", "KoFIU", "STR", "특금법",
    ])
    def test_is_valid_true_when_keyword_in_draft(self, keyword):
        """법령 키워드가 SAR 초안에 있으면 is_valid == True."""
        cb, rr = _make_mocks(
            stream_tokens=[f"보고서 내용 {keyword} 위반 사항"],
            finalize_return=f"최종 보고서 {keyword}",
        )
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["is_valid"] is True

    def test_is_valid_false_when_no_keyword(self):
        """법령 키워드가 없으면 is_valid == False."""
        cb, rr = _make_mocks(
            stream_tokens=["일반적인 보고서 내용입니다. 키워드 없음."],
            finalize_return="최종 보고서",
        )
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["is_valid"] is False

    def test_finalize_called_with_draft_and_payload(self):
        """validate 노드가 report_runner.finalize 를 올바른 인자로 호출한다."""
        tokens = ["STR 보고서"]
        cb, rr = _make_mocks(stream_tokens=tokens)
        graph  = _build(cb, rr)
        state  = _base_input()
        graph.invoke(state)
        call_args = rr.finalize.call_args[0]
        assert call_args[0] == "STR 보고서"
        assert call_args[1] == state["sar_payload"]

    def test_final_report_from_finalize(self):
        """final_report 는 report_runner.finalize() 반환값이다."""
        cb, rr = _make_mocks(finalize_return="완성된 SAR 특금법")
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())
        assert result["final_report"] == "완성된 SAR 특금법"


# ---------------------------------------------------------------------------
# 5b. LangSmith 피드백 로깅
# ---------------------------------------------------------------------------

class TestLangSmithFeedback:
    def test_validate_calls_feedback_hook(self):
        """validate 노드가 _log_validation_feedback 를 결과와 함께 호출한다."""
        with patch("reporters.sar_graph._log_validation_feedback") as mock_fb:
            cb, rr = _make_mocks()
            graph  = _build(cb, rr)
            result = graph.invoke(_base_input())
            mock_fb.assert_called_once_with(result["is_valid"], result["query_type"])

    def test_feedback_skipped_when_tracing_disabled(self, monkeypatch):
        """LANGCHAIN_TRACING_V2 미설정 시 네트워크 호출 없이 조용히 반환한다."""
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        from reporters.sar_graph import _log_validation_feedback
        with patch("langsmith.Client") as mock_client:
            _log_validation_feedback(True, "network")
            mock_client.assert_not_called()

    def test_feedback_never_raises_outside_trace_context(self, monkeypatch):
        """트레이싱 켜져 있어도 트레이스 컨텍스트 밖이면 예외 없이 스킵한다."""
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        from reporters.sar_graph import _log_validation_feedback
        # get_current_run_tree() 가 None → create_feedback 미호출, 예외 없음
        _log_validation_feedback(False, "standard")


# ---------------------------------------------------------------------------
# 6. End-to-end integration (mocked)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_all_state_fields_populated(self):
        """전체 파이프라인 실행 후 AMLState 핵심 필드가 모두 채워진다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)
        result = graph.invoke(_base_input())

        assert result["query_type"]         in ("network", "standard")
        assert isinstance(result["rag_scores"], list)
        assert isinstance(result["rag_context"], str)
        assert isinstance(result["graph_context"], str)
        assert isinstance(result["integrated_context"], str)
        assert isinstance(result["sar_draft"], str)
        assert isinstance(result["final_report"], str)
        assert isinstance(result["is_valid"], bool)

    def test_graph_is_reusable(self):
        """같은 graph 인스턴스를 두 번 invoke 해도 결과가 독립적이다."""
        cb, rr = _make_mocks()
        graph  = _build(cb, rr)

        r1 = graph.invoke(_base_input())
        r2 = graph.invoke(_base_input())

        assert r1["query_type"] == r2["query_type"]
        assert r1["final_report"] == r2["final_report"]
        # 두 번째 실행에서도 LLM 토큰 수집이 정상 동작해야 한다
        assert r2["sar_draft"] == r1["sar_draft"]
        assert r2["sar_draft"] != ""
        assert r2["is_valid"] == r1["is_valid"]
