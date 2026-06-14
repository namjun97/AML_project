"""Unit tests for reporters/ai_report_generator.py — Groq TPM 재시도 로직.

외부 의존성(Groq API) 없이 가짜 클라이언트/예외로 재시도 동작만 검증한다.
time.sleep 은 패치하여 실제 대기 없이 즉시 실행된다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reporters.ai_report_generator import (
    _completion_with_retry,
    _is_rate_limit_error,
    _retry_after_seconds,
    _MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# 가짜 예외/응답
# ---------------------------------------------------------------------------

class _FakeRateLimitError(Exception):
    """groq.RateLimitError 를 흉내낸다 (클래스명으로 판별됨)."""
    __name__ = "RateLimitError"

    def __init__(self, retry_after: str | None = None):
        super().__init__("rate limit exceeded")
        if retry_after is not None:
            self.response = MagicMock()
            self.response.headers = {"retry-after": retry_after}


# 클래스명이 'RateLimitError' 가 되도록 강제 (판별 로직이 __class__.__name__ 사용)
_FakeRateLimitError.__name__ = "RateLimitError"
_FakeRateLimitError.__qualname__ = "RateLimitError"


class _Status429Error(Exception):
    status_code = 429


class _OtherError(Exception):
    """재시도 대상이 아닌 일반 예외."""


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------

class TestIsRateLimitError:
    def test_detects_by_class_name(self):
        assert _is_rate_limit_error(_FakeRateLimitError()) is True

    def test_detects_by_status_code(self):
        assert _is_rate_limit_error(_Status429Error()) is True

    def test_other_error_not_rate_limit(self):
        assert _is_rate_limit_error(_OtherError()) is False


# ---------------------------------------------------------------------------
# _retry_after_seconds
# ---------------------------------------------------------------------------

class TestRetryAfterSeconds:
    def test_uses_header_when_present(self):
        exc = _FakeRateLimitError(retry_after="12")
        assert _retry_after_seconds(exc, default=99.0) == 12.0

    def test_falls_back_to_default(self):
        assert _retry_after_seconds(_OtherError(), default=7.0) == 7.0

    def test_invalid_header_falls_back(self):
        exc = _FakeRateLimitError(retry_after="not-a-number")
        assert _retry_after_seconds(exc, default=5.0) == 5.0


# ---------------------------------------------------------------------------
# _completion_with_retry
# ---------------------------------------------------------------------------

class TestCompletionWithRetry:
    def test_succeeds_first_try_no_sleep(self):
        client = MagicMock()
        client.chat.completions.create.return_value = "OK"
        with patch("reporters.ai_report_generator.time.sleep") as sleep:
            out = _completion_with_retry(client, model="m", messages=[])
        assert out == "OK"
        sleep.assert_not_called()
        assert client.chat.completions.create.call_count == 1

    def test_retries_then_succeeds(self):
        """첫 호출 429 → 대기 후 재시도 성공."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _FakeRateLimitError(), "OK",
        ]
        with patch("reporters.ai_report_generator.time.sleep") as sleep:
            out = _completion_with_retry(client, model="m", messages=[])
        assert out == "OK"
        assert sleep.call_count == 1
        assert client.chat.completions.create.call_count == 2

    def test_raises_after_exhausting_retries(self):
        """모든 재시도가 429 면 마지막에 예외를 전파한다."""
        client = MagicMock()
        client.chat.completions.create.side_effect = _FakeRateLimitError()
        with patch("reporters.ai_report_generator.time.sleep"):
            with pytest.raises(Exception) as ei:
                _completion_with_retry(client, model="m", messages=[])
        assert ei.value.__class__.__name__ == "RateLimitError"
        # 최초 1회 + _MAX_RETRIES 회
        assert client.chat.completions.create.call_count == _MAX_RETRIES + 1

    def test_non_rate_limit_error_not_retried(self):
        """429 가 아닌 예외는 즉시 전파 (재시도/대기 없음)."""
        client = MagicMock()
        client.chat.completions.create.side_effect = _OtherError("boom")
        with patch("reporters.ai_report_generator.time.sleep") as sleep:
            with pytest.raises(_OtherError):
                _completion_with_retry(client, model="m", messages=[])
        sleep.assert_not_called()
        assert client.chat.completions.create.call_count == 1

    def test_honors_retry_after_header(self):
        """Retry-After 헤더 값이 sleep 대기 시간으로 사용된다."""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _FakeRateLimitError(retry_after="3"), "OK",
        ]
        with patch("reporters.ai_report_generator.time.sleep") as sleep:
            _completion_with_retry(client, model="m", messages=[])
        sleep.assert_called_once_with(3.0)


# ---------------------------------------------------------------------------
# ReportRunner.finalize — 기술 용어(임베딩) 평이화 치환
# ---------------------------------------------------------------------------

class TestSarTermReplacement:
    """SAR 본문의 '임베딩' 계열 기술 용어가 수사관 친화 표현으로 치환되는지 검증."""

    def _runner(self):
        from reporters.report_runner import ReportRunner
        return ReportRunner(MagicMock())

    def _payload(self):
        return {"report_context": {
            "target_node_id": 1, "risk_level": "고위험(High)",
            "fraud_probability": "99.78%", "analysis_date": "2026-06-14",
        }}

    def test_embedding_similarity_replaced(self):
        rr = self._runner()
        draft = "##II##\n거래 그래프 임베딩 유사성에 기반한 점수\n##II_END##"
        out = rr.finalize(draft, self._payload(), graph_context="x")
        assert "임베딩 유사성" not in out
        assert "거래 행태·자금 흐름 구조의 유사성" in out

    def test_gnn_embedding_replaced(self):
        rr = self._runner()
        draft = "##III##\nGNN 임베딩 분석 결과\n##III_END##"
        out = rr.finalize(draft, self._payload(), graph_context="x")
        assert "GNN 임베딩" not in out

    def test_graph_context_interpretation_preserved(self):
        """V장 GraphRAG 해석 블록(graph_context)은 '임베딩' 정의를 보존한다."""
        rr = self._runner()
        gctx = "수치 지문(임베딩)으로 요약한 결과"
        out = rr.finalize("##II##\n내용\n##II_END##", self._payload(), graph_context=gctx)
        assert "수치 지문(임베딩)" in out  # 정의 목적의 임베딩 표기는 유지


# ---------------------------------------------------------------------------
# _extract_section — _END 마커 누락 견고성
#
# llama-3.1-8b 는 ##II##/##III##/##IV## 시작 마커만 찍고 _END 마커를 종종
# 생략한다. 특히 마지막 ##IV## 는 뒤에 마커가 없어, _END 를 필수로 요구하면
# IV 가 매번 '파싱 실패'로 떨어진다. 시작 마커 ~ 다음 마커/EOS 로 닫혀야 한다.
# ---------------------------------------------------------------------------

class TestExtractSectionWithoutEndMarker:
    _RAW = (
        "##II## \n1. 첫째 징후.\n2. 둘째 징후.\n"
        "##III## \n위험 평가 본문.\n"
        "##IV## \n1. 첫 조치.\n5. 특정금융정보법 제4조에 따라 STR 을 즉시 제출한다."
    )

    def _ex(self, start, end):
        from reporters.sar_template import _extract_section
        return _extract_section(self._RAW, start, end)

    def test_ii_closed_by_next_marker(self):
        out = self._ex("##II##", "##II_END##")
        assert "첫째 징후" in out and "둘째 징후" in out
        assert "위험 평가" not in out  # III 로 새지 않음

    def test_iii_closed_by_next_marker(self):
        out = self._ex("##III##", "##III_END##")
        assert out.strip() == "위험 평가 본문."

    def test_iv_closed_by_eos(self):
        """마지막 IV 는 뒤에 마커가 없어도 문자열 끝까지 잡혀 완결돼야 한다."""
        out = self._ex("##IV##", "##IV_END##")
        assert out.rstrip().endswith("제출한다.")
        assert "5. 특정금융정보법" in out

    def test_explicit_end_marker_still_works(self):
        from reporters.sar_template import _extract_section
        raw = "##II##\n내용만.\n##II_END##\n뒤쪽 잡소리"
        out = _extract_section(raw, "##II##", "##II_END##")
        assert out.strip() == "내용만."
        assert "잡소리" not in out

    def test_no_parsing_failure_fallback(self):
        """assemble_sar_template 가 _END 누락 출력에도 폴백 문구를 내지 않는다."""
        from reporters.sar_template import assemble_sar_template
        payload = {"report_context": {
            "target_node_id": 7, "risk_level": "고위험(High)",
            "fraud_probability": "99.96%", "analysis_date": "2026-06-14",
        }}
        out = assemble_sar_template(self._RAW, payload, graph_context="네트워크 분석")
        assert "파싱 실패" not in out
        assert "제출한다." in out
