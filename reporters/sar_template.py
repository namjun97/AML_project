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

[II. 주요 의심 거래 징후 — 작성 지침]
- 3~5개의 번호 매긴 항목으로 간결하게 작성하십시오 (한 문단으로 길게 나열하지 마십시오).
- 거래 네트워크 분석의 '행동 신호 (기저율 대비 변별력)'에서 **변별력이 '보통' 또는 '높음'인
  신호만** 의심 근거로 제시하십시오. **변별력 '낮음' 신호는 의심 근거로 쓰지 마십시오**
  (전체 계좌에 흔해 사기를 구분하지 못하는 신호임). 언급이 불가피하면 "흔한 패턴으로 단독
  변별력은 낮음"이라고 정직하게 표기하십시오. 절대 흔한 신호를 '자금 은닉/유출 정황'으로 단정하지 마십시오.
- 변별력 있는 신호가 없으면, 의심의 1차 근거가 모델 의심도임을 밝히되, '임베딩 유사성'이라는
  기술 용어를 그대로 쓰지 말고 수사관이 이해할 수 있게 풀어 설명하십시오. 즉 "이 계좌의
  거래 행태와 자금 흐름 구조(거래 규모·입출금 방향·거래 상대 구성·자금 경로 형태)가 과거
  확인된 자금세탁 계좌들과 통계적으로 유사하다"는 의미이며, 특정 거래를 적발한 직접 증거가
  아니라 '유형 유사성에 기반한 검토 대상 선별 신호'임을 명시하십시오.
- '모델 의심도 등급(고위험 등)'을 "고위험으로 분류됨"처럼 수사 확정 결론으로 서술하지 마십시오.
  그것은 자동 선별 점수일 뿐입니다.
- 위험도가 낮음을 뜻하는 정상 수치(0%대 클러스터 위험도, 사기 경로 없음 등)는 의심 사유로 쓰지 마십시오.
- GNN 임베딩 개별 차원(GNN_Emb_n)은 나열하지 마십시오.

[III. 자금세탁 위험 평가 — 작성 지침]
- 거래 네트워크 분석의 '종합 수사 평가'에 제시된 '수사 우선순위 권고'와 '판단'을 그대로
  반영하십시오. **이 우선순위와 일관된 결론만 쓰고, 서로 모순되는 문장을 섞지 마십시오.**
  · 우선순위가 '고위험'이면: 확인된 근거(세탁 거래/네트워크 연계/모티프)를 들어 고위험으로 기술하고,
    "1차 의심 후보"·"단정하지 않음" 같은 약화 표현을 쓰지 마십시오.
  · 우선순위가 '검토 필요(1차 의심 후보)'이면: 근거 부족을 인정하고 "추가 인적 검토 필요"로 신중히
    기술하며 "고위험 확정"·"네트워크상 위험한 위치" 단정을 내리지 마십시오.
- '모델 의심도'를 설명할 땐 '임베딩 유사성' 용어 대신 "거래 행태·자금 흐름 구조가 과거
  자금세탁 계좌와 유사함"이라는 평이한 해석으로 기술하십시오.
- 본 문서는 특정금융정보법 제4조 '의심거래 보고' 대상 여부 판단 보고서입니다.
  "법을 위반했다/하지 않았다"는 단정 대신 "의심거래 보고 대상에 해당한다" 수준으로 기술하십시오.
- 2~3문장으로 평가만 요약하고, II·V장 수치를 다시 나열하지 마십시오.

[IV. 조치 권고 사항 — 작성 지침]
- 거래 모니터링 강화, 계좌 거래내역 정밀 조사, 필요 시 KoFIU 보고 등 실행 가능한 조치를
  구체적으로 기술하십시오.

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
