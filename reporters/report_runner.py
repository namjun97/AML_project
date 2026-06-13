from __future__ import annotations

from typing import Callable, Generator

from reporters.ai_report_generator import (
    _strip_rag_from_output,
    stream_ai_report,
    generate_ai_report,
)
from reporters.context_builder import ContextBuilder
from reporters.sar_template import assemble_sar_template


# LLM 본문(II·III·IV)의 기술 용어 → 수사관 친화 평이 표현 치환 (긴 표현 우선).
# 모델 점수의 근거가 '거래 행태·자금 흐름 구조 유사성'임을 보고서 전반에서 일관 유지.
_SAR_TERM_REPLACEMENTS = [
    ("거래 그래프 임베딩 유사성", "거래 행태·자금 흐름 구조의 유사성"),
    ("그래프 임베딩 유사성",     "거래 행태·자금 흐름 구조의 유사성"),
    ("임베딩 유사성",            "거래 행태·자금 흐름 구조의 유사성"),
    ("거래 그래프 임베딩",       "거래 행태·자금 흐름 구조"),
    ("GNN 임베딩",              "거래 행태·자금 흐름 구조"),
]


class ReportRunner:
    """
    SAR 보고서 생성 파이프라인 오케스트레이터.

    컨텍스트 조회(ContextBuilder) → Groq 스트리밍 → 양식 조립
    (assemble_sar_template)의 3단계를 캡슐화합니다.

    Attributes:
        context_builder (ContextBuilder): RAG·GraphRAG 컨텍스트 조회 인스턴스
        model           (str)           : Groq 모델명

    Example:
        runner = ReportRunner(context_builder)

        # 스트리밍 모드 (Streamlit 권장)
        rag_ctx, graph_ctx = runner.build_contexts(sar_payload, node_idx)
        for token in runner.stream(sar_json_str, rag_ctx, graph_ctx):
            raw_text += token
        final_report = runner.finalize(raw_text, sar_payload, graph_ctx)

        # 원스텝 비스트리밍 모드 (폴백)
        final_report, rag_ctx, graph_ctx = runner.run(
            sar_json_str, sar_payload, node_idx
        )

    Groq 모델 선택 가이드:
        - "llama-3.1-8b-instant"    : 기본값. 빠르고 무료 한도 넉넉
        - "llama-3.3-70b-versatile" : 더 정교한 보고서가 필요할 때
    """

    DEFAULT_MODEL = "llama-3.1-8b-instant"

    def __init__(
        self,
        context_builder: ContextBuilder,
        model: str = DEFAULT_MODEL,
        # 하위 호환 인자 (무시됨 — Groq SDK가 연결을 관리)
        ollama_url: str | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.model = model

    # ------------------------------------------------------------------
    # 1단계: 컨텍스트 조회
    # ------------------------------------------------------------------

    def build_contexts(
        self, sar_payload: dict, node_idx: int
    ) -> tuple[str, str]:
        """
        텍스트 RAG + GraphRAG 컨텍스트를 조회합니다.

        Returns:
            tuple[str, str]: (rag_context, graph_context)
        """
        return self.context_builder.build_all(sar_payload, node_idx)

    # ------------------------------------------------------------------
    # 2단계: Groq 스트리밍
    # ------------------------------------------------------------------

    def stream(
        self,
        sar_json_str: str,
        rag_context: str,
        graph_context: str,
    ) -> Generator[str, None, None]:
        """
        Groq Streaming API로 SAR 섹션(II~IV)을 토큰 단위로 생성합니다.

        Yields:
            str: 토큰 단위 텍스트 조각
        """
        yield from stream_ai_report(
            json_data     = sar_json_str,
            model         = self.model,
            rag_context   = rag_context,
            graph_context = graph_context,
        )

    # ------------------------------------------------------------------
    # 3단계: 후처리 + 양식 조립
    # ------------------------------------------------------------------

    def finalize(
        self,
        raw_text: str,
        sar_payload: dict,
        graph_context: str,
    ) -> str:
        """
        스트리밍으로 수집된 raw 텍스트를 정제하고 고정 SAR 양식에 조립합니다.

        처리 순서:
            1. RAG 에코 제거 (_strip_rag_from_output)
            2. 언어 말투 보정
            3. assemble_sar_template() 호출 (Section V에 graph_context 삽입)

        Returns:
            str: 완성된 SAR 보고서 텍스트
        """
        clean = _strip_rag_from_output(raw_text)
        clean = clean.replace("があります", "가 있습니다")
        clean = clean.replace("必要があります", "필요가 있습니다")
        # '임베딩 유사성' 등 기술 용어를 수사관 친화 평이 표현으로 결정적 치환.
        # (8B LLM 이 프롬프트의 용어 금지 지시를 완전히 따르지 않을 때 대비.
        #  V장 GraphRAG 해석 블록은 graph_context 로 별도 삽입돼 영향받지 않음.)
        for _jargon, _plain in _SAR_TERM_REPLACEMENTS:
            clean = clean.replace(_jargon, _plain)
        return assemble_sar_template(clean, sar_payload, graph_context=graph_context)

    # ------------------------------------------------------------------
    # 원스텝 비스트리밍 실행 (폴백용)
    # ------------------------------------------------------------------

    def run(
        self,
        sar_json_str: str,
        sar_payload: dict,
        node_idx: int,
        on_token: Callable[[str], None] | None = None,
    ) -> tuple[str, str, str]:
        """
        컨텍스트 조회 → 스트리밍 → 조립을 원스텝으로 실행합니다.

        Args:
            sar_json_str (str)          : SAR JSON 문자열
            sar_payload  (dict)         : SAR 페이로드
            node_idx     (int)          : 분석 대상 노드 인덱스
            on_token     (callable|None): 토큰 수신 콜백 (선택)

        Returns:
            tuple[str, str, str]: (final_report, rag_context, graph_context)
        """
        rag_ctx, graph_ctx = self.build_contexts(sar_payload, node_idx)

        raw_text = ""
        for token in self.stream(sar_json_str, rag_ctx, graph_ctx):
            raw_text += token
            if on_token:
                on_token(token)

        final_report = self.finalize(raw_text, sar_payload, graph_ctx)
        return final_report, rag_ctx, graph_ctx
