from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# 고정 양식 정의
_W  = 70            # 구분선 너비
_EQ = "═" * _W      # 두꺼운 구분선
_DH = "─" * _W      # 얇은 구분선

SAR_TEMPLATE = (
    "\n"
    + _EQ + "\n"
    + "         의심거래보고서 (Suspicious Activity Report)\n"
    + _EQ + "\n"
    + "문서 번호  : SAR-{year}-{node_id:06d}\n"
    + "보안 등급  : 대외비 (CONFIDENTIAL)\n"
    + "작성 일자  : {date}\n"
    + "보고 근거  : 특정 금융거래정보의 보고 및 이용 등에 관한 법률 제4조\n"
    + _EQ + "\n"
    + "\n"
    + "I. 분석 개요 (Analysis Overview)\n"
    + _DH + "\n"
    + "□ 분석 대상 계좌 ID    : {node_id}\n"
    + "□ 모델 의심도 등급     : {risk_level}  (자동 선별 — 수사 결론은 III·V장 참조)\n"
    + "□ 모델 사기 의심 확률  : {fraud_prob}\n"
    + "□ 분석 기법            : GNN(GraphSAGE) + XGBoost 하이브리드 모델\n"
    + "□ 분석 일자          : {date}\n"
    + "\n"
    + "II. 주요 의심 거래 징후 (Key Suspicious Indicators)\n"
    + _DH + "\n"
    + "{section_ii}\n"
    + "\n"
    + "III. 자금세탁 위험 평가 (AML Risk Assessment)\n"
    + _DH + "\n"
    + "{section_iii}\n"
    + "\n"
    + "IV. 조치 권고 사항 (Recommended Actions)\n"
    + _DH + "\n"
    + "{section_iv}\n"
    + "\n"
    + "V. 거래 네트워크 분석 (Transaction Network Analysis)\n"
    + _DH + "\n"
    + "{section_v}\n"
    + "\n"
    + _EQ + "\n"
    + "※ 본 보고서는 AI 기반 자동 분석 시스템에 의해 생성되었으며,\n"
    + "  담당자의 최종 검토 및 서명이 필요합니다.\n"
    + "작성 시스템  : KoFIU AML 분석 시스템 (GNN + XGBoost + RAG + GraphRAG)\n"
    + "생성 일시    : {timestamp}\n"
    + _EQ + "\n"
)

# Neo4j 미연결 시 Section V에 삽입할 기본 메시지
_SECTION_V_UNAVAILABLE = (
    "※ Neo4j 그래프 DB에 연결되지 않아 네트워크 구조 분석을 수행할 수 없습니다.\n"
    "  Neo4j Desktop을 실행하고 데이터를 로드한 후 보고서를 재생성하십시오."
)


# ======================================================================
# LLM 프롬프트용 섹션 형식 지시문
# ======================================================================

SECTION_FORMAT_INSTRUCTIONS = """\
[출력 형식]
- 첫 글자는 반드시 '#' 이어야 하며, 서문·인사말은 출력하지 마십시오.
- 정확히 아래 순서로 세 섹션을 마커로 감싸 출력하고, 각 마커 쌍 사이에는 해당 섹션의
  '완성된 본문 문장'만 작성하십시오:
      ##II##  …본문…  ##II_END##
      ##III## …본문…  ##III_END##
      ##IV##  …본문…  ##IV_END##
- 마커·지침 문구·괄호 설명·JSON 필드명(behavioral_factors, 주요_판단_근거, network_signal,
  지표, 기여방향 등)을 본문에 절대 복사하지 마십시오. 데이터 값을 읽어 자연스러운 한국어
  완성 문장으로 풀어 쓰십시오.

[공통 — 반복 절대 금지]
- 세 섹션(II·III·IV)에 같은 문장이나 같은 표현을 반복하지 마십시오. 각 항목은 **서로 다른
  내용**을 담아야 합니다. 특히 "거래 행태가 과거 자금세탁 계좌와 유사함" 같은 문장을 여러 번
  되풀이하지 마십시오 — 그 취지는 III장에서 한 번만 언급합니다.
- 자금세탁 조사 실무자에게 보고하듯, 간결하고 단정적인 수사 보고체로 작성하십시오.

[II. 주요 의심 거래 징후 — 작성 지침]
- 거래 네트워크 분석의 '핵심 의심 근거(요약)' 항목들을 **각각 서로 다른 번호 항목**으로 풀어
  쓰되, 막연한 표현이 아니라 **구체적 수치·계좌번호·건수**를 포함하십시오.
  예: "5개 계좌로 분산 송금하여 전체 평균(0.9개)의 5.5배에 달하는 팬아웃 구조를 보임",
      "자금세탁 확정 계좌 1731585361과 직접 거래가 확인됨",
      "수취 거래 중 1건이 자금세탁으로 확정된 거래임".
- '변별력 낮음' 신호나 정상 수치(평균 이하 등)는 의심 근거로 쓰지 마십시오.
- 모델 의심도를 인용할 땐 '임베딩 유사성' 용어 대신 "거래 행태·자금 흐름 구조가 과거 자금세탁
  계좌와 유사함"으로 풀어 쓰고, 이는 검토 대상 선별 신호임을 밝히십시오.

[III. 자금세탁 위험 평가 — 작성 지침]
- '종합 수사 평가'의 '수사 우선순위 권고'와 '판단'을 일관되게 반영해 **2~3문장**으로 종합
  의견을 기술하십시오. 우선순위와 모순되는 문장을 섞지 마십시오.
  · 우선순위가 '고위험'이면 확인된 근거를 들어 단정적으로 기술하고 "1차 후보"·"단정 안 함" 같은
    약화 표현을 쓰지 마십시오. '검토 필요'이면 근거 부족을 인정하고 신중히 기술하십시오.
- 특정금융정보법 제4조 '의심거래 보고' 대상 여부 판단 보고서이므로, "법 위반" 단정 대신
  "의심거래 보고 대상에 해당한다" 수준으로 기술하십시오. II·V장 수치를 다시 나열하지 마십시오.

[IV. 조치 권고 사항 — 작성 지침]
- 거래 네트워크 분석의 '권고 조치(우선순위 반영)' 목록을 **각 항목별로 서로 다른 실행 조치**로
  풀어 쓰십시오. 각 조치는 "무엇을, 왜" 하는지 한 문장으로 구체적으로 기술합니다
  (예: 거래 일시 제한, 원본 거래내역 확보, 연결 계좌 동반 조사, STR 제출 등).
- 위험 평가 문장(III장 내용)을 조치 항목에 반복하지 마십시오 — IV장은 '할 일'만 적습니다.

[엄격 규칙]
- 마커(##II## 등)는 줄 맨 앞에 단독으로 두고, **·*·[]·() 같은 마크다운 기호를 붙이지 마십시오
- ##II## 이전에 어떠한 텍스트도 출력하지 마십시오
- 각 섹션은 개조식(1., 2., 가.)의 완성된 한국어 문장으로 작성하십시오
- 지침 문구·괄호 설명·JSON 필드명·영문 키를 본문에 복사하지 마십시오
- '임베딩', '임베딩 유사성', '벡터', 'GNN 임베딩' 같은 기술 용어를 단독으로 쓰지 마십시오.
  반드시 "이 계좌의 거래 행태·자금 흐름 구조가 과거 확인된 자금세탁 계좌와 통계적으로 유사함"
  같은 평이한 수사 언어로 풀어 쓰십시오.
- 제공된 데이터에 없는 수치나 법 위반 사실을 지어내지 마십시오
"""

# 내부 유틸

def _safe_section(text: str) -> str:
    """
    str.format() 호출 시 LLM 출력의 중괄호({, })가 포맷 지시자로 해석되어
    KeyError 또는 ValueError가 발생하는 것을 방지합니다.

    Python의 str.format()은 {{ → {, }} → } 로 리터럴 변환하므로,
    LLM 출력 내의 중괄호를 이중 중괄호로 이스케이프합니다.
    """
    return text.replace("{", "{{").replace("}", "}}")


def _normalize_markers(text: str) -> str:
    """
    LLM이 섹션 마커를 마크다운 변형으로 출력한 경우 정규화합니다.

    처리 대상 변형 예시:
        **##II##**   → ##II##
        ** ##II## ** → ##II##
        ## II ##     → ##II##
        [##II##]     → ##II##
        `##II##`     → ##II##
    """
    # 1) 마커 앞뒤의 마크다운 기호 제거 (* ** ` [ ] ( ))
    text = re.sub(r'[*`\[\]()]+\s*(##\w+##)\s*[*`\[\]()]+', r'\1', text)
    # 2) ## 내부 공백 제거: ## II ## → ##II##
    text = re.sub(r'##\s+(\w+)\s+##', r'##\1##', text)
    # 3) 줄 앞쪽의 공백+마크다운 헤더(###, ##) 제거 후 마커만 남김
    #    예: "### ##II##" → "##II##"
    text = re.sub(r'^[ \t]*#{1,3}[ \t]+(##\w+##)', r'\1', text, flags=re.MULTILINE)
    return text


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:

    pattern = re.compile( # start_marker - end_marker 사이 내용 추출
        re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
        re.DOTALL | re.IGNORECASE,
    )

    # 1차 시도: 마커 정규화 후 추출 → 내용에 ** / `` 잔류 없음
    normalized = _normalize_markers(text)
    match = pattern.search(normalized)
    if match:
        return match.group(1).strip()

    # 2차 시도: 원문에서 직접 추출 (폴백)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # 3차 시도: end_marker 가 없을 때 — 시작 마커 ~ '다음 ## 마커 또는 문자열 끝'.
    #   LLM(llama-3.1-8b)이 _END 마커를 종종 생략하고 다음 시작 마커(##III## 등)만
    #   구분자로 쓴다. 특히 마지막 ##IV## 는 뒤에 마커가 없어 EOS 로 닫아야 한다.
    #   (이 폴백이 없으면 IV 가 매번 '파싱 실패'로 떨어졌음)
    open_pat = re.compile(
        re.escape(start_marker) + r"(.*?)(?=##\s*[IVXivx]{1,3}(?:_END)?\s*##|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for candidate in (normalized, text):
        m = open_pat.search(candidate)
        if m and m.group(1).strip():
            return m.group(1).strip()

    return ""


def _fallback_section(raw_text: str, section_label: str) -> str: # 마커 파싱에 실패 시 폴백 텍스트 반환
    header_pat = re.compile(
        rf"{re.escape(section_label)}[^\n]*\n(.*?)(?=\n[IVX]{{1,3}}\.|$)",
        re.DOTALL,
    )
    m = header_pat.search(raw_text)
    if m:
        return m.group(1).strip()
    return "(AI 생성 내용 파싱 실패 — 원문을 참고하십시오)\n\n" + raw_text[:600]


# 조립 함수

def assemble_sar_template(
    raw_llm_output: str,
    sar_payload: dict[str, Any],
    graph_context: str = "",
) -> str:
    ctx = sar_payload.get("report_context", {})
    now = datetime.now()

    # ── 섹션 추출 (II·III·IV : LLM 생성) ─────────────────────────────
    sec_ii  = _extract_section(raw_llm_output, "##II##",  "##II_END##")
    sec_iii = _extract_section(raw_llm_output, "##III##", "##III_END##")
    sec_iv  = _extract_section(raw_llm_output, "##IV##",  "##IV_END##")

    # 마커 없이 원문이 나온 경우 폴백
    if not sec_ii:
        sec_ii  = _fallback_section(raw_llm_output, "II.")
    if not sec_iii:
        sec_iii = _fallback_section(raw_llm_output, "III.")
    if not sec_iv:
        sec_iv  = _fallback_section(raw_llm_output, "IV.")

    # ── Section V : GraphRAG 결과 직접 삽입 ───────────────────────────
    sec_v = graph_context.strip() if graph_context.strip() else _SECTION_V_UNAVAILABLE

    # ── str.format() 충돌 방지: 중괄호 이스케이프 ─────────────────────
    sec_ii  = _safe_section(sec_ii)
    sec_iii = _safe_section(sec_iii)
    sec_iv  = _safe_section(sec_iv)
    sec_v   = _safe_section(sec_v)

    # ── 양식 조립 ──────────────────────────────────────────────────────
    node_id    = ctx.get("target_node_id", 0)
    risk_level = ctx.get("risk_level", "—")
    fraud_prob = ctx.get("fraud_probability", "—")
    date_str   = ctx.get("analysis_date", now.strftime("%Y-%m-%d"))

    return SAR_TEMPLATE.format(
        year        = now.year,
        node_id     = node_id,
        date        = date_str,
        risk_level  = risk_level,
        fraud_prob  = fraud_prob,
        section_ii  = sec_ii,
        section_iii = sec_iii,
        section_iv  = sec_iv,
        section_v   = sec_v,
        timestamp   = now.strftime("%Y-%m-%d %H:%M:%S"),
    )
