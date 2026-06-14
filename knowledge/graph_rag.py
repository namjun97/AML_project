from __future__ import annotations

import warnings
from typing import Any

from neo4j import GraphDatabase


# GraphRAGRetriever — SAML-D 거래 네트워크 기반 GraphRAG 컨텍스트 생성기
#
# SAML-D 는 실제 자금세탁 유형(팬인/팬아웃·레이어링·환치기·역외이전 등)을 담은
# 합성 AML 데이터셋이라, PaySim 과 달리 '네트워크 증거'가 실제로 존재한다.
# 노드 속성(loader 기준): account_id, node_idx, is_laundering, fraud_prob,
#   out_count, in_count, unique_receivers, unique_senders, out_amount, in_amount,
#   cross_border_ratio, currency_mismatch_ratio, passthrough_ratio
# 엣지 속성: amount, payment_type, sender_location, receiver_location,
#   cross_border, currency_mismatch, is_laundering_tx, laundering_type, date

class GraphRAGRetriever:
    def __init__(
        self,
        uri: str      = "bolt://localhost:7687",
        user: str     = "neo4j",
        password: str = "qwer1234",
        max_paths: int = 5,
        lazy: bool    = False,
    ) -> None:
        self._uri      = uri
        self._user     = user
        self._password = password
        self._max_paths = max_paths
        self._driver   = None
        self._available = False
        self._connect_attempted = False
        self._pop_stats = None  # 기저율(base-rate) 캐시 — 모티프 변별력 계산용

        if not lazy:
            self._ensure_connected()

    # ------------------------------------------------------------------
    # 연결 관리
    # ------------------------------------------------------------------
    def _try_connect(self) -> None:
        driver = None
        try:
            driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password),
                max_connection_pool_size=5,
                connection_acquisition_timeout=30, liveness_check_timeout=2,
            )
            driver.verify_connectivity()
        except Exception as e:
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass
            self._available = False
            warnings.warn(f"[GraphRAG] Neo4j 연결 실패 — 그래프 RAG 비활성화: {e}",
                          RuntimeWarning, stacklevel=2)
            return
        self._driver = driver
        self._available = True
        try:
            print("[GraphRAG] Neo4j 연결 성공")
        except Exception:
            pass

    def _ensure_connected(self) -> bool:
        if not self._connect_attempted:
            self._connect_attempted = True
            self._try_connect()
        return self._available

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def connect_attempted(self) -> bool:
        return self._connect_attempted

    @property
    def is_configured(self) -> bool:
        return bool(self._uri and self._password)

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def _run(self, query: str, **params) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            return [dict(r) for r in session.run(query, **params)]

    @staticmethod
    def _nz(v, default=0):
        return default if v is None else v

    @staticmethod
    def _won(v) -> str:
        a = float(v or 0)
        if a >= 1_000_000:
            return f"{a/1_000_000:.2f}M"
        if a >= 1_000:
            return f"{a:,.0f}"
        return f"{a:.0f}"

    # ------------------------------------------------------------------
    # 기저율(base-rate) + 모티프 변별력
    # ------------------------------------------------------------------
    # 모티프 신호 정의: (라벨, Cypher 조건식, 약어키)
    _SIGNALS = [
        ("다수 계좌로 분산 송금(팬아웃)",        "a.unique_receivers >= 5",        "fo"),
        ("다수 계좌로부터 자금 집결(팬인)",       "a.unique_senders >= 5",          "fi"),
        ("받은 자금을 곧바로 송금(경유·레이어링)", "a.passthrough_ratio >= 0.2",     "pt"),
        ("국가 간 거래(역외 이전)",              "a.cross_border_ratio > 0",       "xb"),
        ("송·수취 통화 불일치(환치기)",          "a.currency_mismatch_ratio > 0",  "cm"),
    ]

    def _population_stats(self) -> dict:
        if self._pop_stats is not None:
            return self._pop_stats
        parts = ["count(a) AS total",
                 "sum(CASE WHEN a.is_laundering THEN 1 ELSE 0 END) AS l_total"]
        for _, cond, key in self._SIGNALS:
            parts.append(f"sum(CASE WHEN {cond} THEN 1 ELSE 0 END) AS {key}_t")
            parts.append(f"sum(CASE WHEN {cond} AND a.is_laundering THEN 1 ELSE 0 END) AS {key}_l")
        rows = self._run("MATCH (a:Account) RETURN " + ", ".join(parts))
        self._pop_stats = rows[0] if rows else {}
        return self._pop_stats

    def _signal_assessment(self, prof: dict) -> tuple:
        """노드 프로파일의 모티프 신호별 변별력(lift) 평가.
        Returns (baseline, [{label,rate,lift,tag}], discriminative_count)."""
        pop = self._population_stats()
        total = self._nz(pop.get("total"), 0)
        baseline = (self._nz(pop.get("l_total"), 0) / total) if total else 0.0

        present = {
            "fo": self._nz(prof.get("unique_receivers")) >= 5,
            "fi": self._nz(prof.get("unique_senders")) >= 5,
            "pt": self._nz(prof.get("passthrough_ratio")) >= 0.2,
            "xb": self._nz(prof.get("cross_border_ratio")) > 0,
            "cm": self._nz(prof.get("currency_mismatch_ratio")) > 0,
        }
        evaluated, disc = [], 0
        for label, _, key in self._SIGNALS:
            if not present.get(key):
                continue
            t = self._nz(pop.get(f"{key}_t"), 0)
            l = self._nz(pop.get(f"{key}_l"), 0)
            rate = (l / t) if t else None
            lift = (rate / baseline) if (rate is not None and baseline) else None
            if lift is not None and lift >= 1.3:
                disc += 1
            tag = "높음" if (lift and lift >= 2.0) else "보통" if (lift and lift >= 1.3) else "낮음"
            evaluated.append({"label": label, "rate": rate, "lift": lift, "tag": tag})
        return baseline, evaluated, disc

    # ------------------------------------------------------------------
    # Cypher 쿼리
    # ------------------------------------------------------------------
    def _q0_profile(self, node_idx: int) -> dict:
        rows = self._run(
            """
            MATCH (a:Account {node_idx: $n})
            CALL { MATCH (t:Account) RETURN count(t) AS total }
            CALL { WITH a MATCH (x:Account) WHERE x.fraud_prob >= a.fraud_prob
                   RETURN count(x) AS rank_pos }
            RETURN a.account_id AS account_id, a.fraud_prob AS fraud_prob,
                   a.is_laundering AS is_laundering,
                   a.out_count AS out_count, a.in_count AS in_count,
                   a.unique_receivers AS unique_receivers, a.unique_senders AS unique_senders,
                   a.out_amount AS out_amount, a.in_amount AS in_amount,
                   a.cross_border_ratio AS cross_border_ratio,
                   a.currency_mismatch_ratio AS currency_mismatch_ratio,
                   a.passthrough_ratio AS passthrough_ratio,
                   total, rank_pos
            """, n=node_idx)
        return rows[0] if rows else {}

    def _q1_counterparties(self, node_idx: int) -> dict:
        rows = self._run(
            """
            MATCH (c:Account {node_idx: $n})
            OPTIONAL MATCH (c)-[ro:SENT_TO]->(o:Account)
            WITH c, count(ro) AS out_tx, count(DISTINCT o) AS out_acc,
                 sum(ro.amount) AS out_amt,
                 sum(CASE WHEN ro.is_laundering_tx THEN 1 ELSE 0 END) AS out_l,
                 collect(DISTINCT CASE WHEN ro.is_laundering_tx THEN ro.laundering_type END) AS lt_o
            OPTIONAL MATCH (i:Account)-[ri:SENT_TO]->(c)
            RETURN out_tx, out_acc, out_amt, out_l, lt_o,
                   count(ri) AS in_tx, count(DISTINCT i) AS in_acc,
                   sum(ri.amount) AS in_amt,
                   sum(CASE WHEN ri.is_laundering_tx THEN 1 ELSE 0 END) AS in_l
            """, n=node_idx)
        return rows[0] if rows else {}

    def _q2_launder_neighbors(self, node_idx: int) -> dict:
        """직접 연결된 세탁 계좌 (1-hop) — 빠르고 신뢰도 높은 네트워크 증거."""
        rows = self._run(
            """
            MATCH (c:Account {node_idx: $n})-[:SENT_TO]-(nb:Account)
            WHERE nb.is_laundering = true AND nb.node_idx <> $n
            RETURN count(DISTINCT nb) AS l_neighbors,
                   collect(DISTINCT nb.account_id)[..5] AS ids
            """, n=node_idx)
        return rows[0] if rows else {}

    def _q3_path_to_launder(self, node_idx: int) -> list[dict]:
        rows = self._run(
            f"""
            MATCH (c:Account {{node_idx: $n}})
            MATCH p = shortestPath((c)-[:SENT_TO*1..3]->(f:Account))
            WHERE f.is_laundering = true AND f.node_idx <> $n
            WITH p, f, length(p) AS plen,
                 [x IN nodes(p) | x.account_id] AS accs
            RETURN plen, f.account_id AS target,
                   round(f.fraud_prob*100)/100.0 AS prob, accs
            ORDER BY plen ASC LIMIT {self._max_paths}
            """, n=node_idx)
        return rows

    def _q4_fanout(self, node_idx: int) -> dict:
        rows = self._run(
            """
            CALL { MATCH (x:Account) RETURN avg(x.unique_receivers) AS avg_fo,
                   avg(x.unique_senders) AS avg_fi }
            MATCH (a:Account {node_idx: $n})
            RETURN a.unique_receivers AS fo, a.unique_senders AS fi,
                   round(avg_fo*100)/100.0 AS avg_fo, round(avg_fi*100)/100.0 AS avg_fi
            """, n=node_idx)
        return rows[0] if rows else {}

    # ------------------------------------------------------------------
    # 포맷터
    # ------------------------------------------------------------------
    def _format_q0(self, d: dict, sig_eval: list, baseline: float) -> str:
        if not d:
            return "  (계좌 정보 없음)"
        prob = self._nz(d.get("fraud_prob"), 0.0)
        total = self._nz(d.get("total"), 0)
        rank = self._nz(d.get("rank_pos"), 0)
        pct = (rank / total * 100) if total else 0.0
        lines = [
            f"  · 모델 의심도(네트워크 학습): {min(prob, 0.9999):.2%}  (전체 {total:,}개 계좌 중 상위 {pct:.2f}%)",
            "    └─ 해석: GraphSAGE 가 이 계좌의 '거래 행태 + 주변 자금 흐름 구조'(팬인/팬아웃,",
            "       경유 비율, 역외·환치기 패턴)를 학습한 결과, 과거 확인된 자금세탁 계좌들과",
            "       구조적으로 유사하다는 의미입니다. 검토 대상 선별 신호이며, 확정은 거래내역 검토 필요.",
            f"  · 거래 활동: 송금 {self._nz(d.get('out_count')):,}건(→{self._nz(d.get('unique_receivers')):,}개 계좌) "
            f"/ 수취 {self._nz(d.get('in_count')):,}건(←{self._nz(d.get('unique_senders')):,}개 계좌)",
            f"  · 금액: 총 송금 ₩{self._won(d.get('out_amount'))} / 총 수취 ₩{self._won(d.get('in_amount'))}",
        ]
        if sig_eval:
            lines.append("  · 자금세탁 모티프 (기저율 대비 변별력):")
            for s in sig_eval:
                if s["lift"] is not None:
                    lines.append(
                        f"    - {s['label']}: 해당 신호 보유 계좌 세탁율 {s['rate']:.1%} "
                        f"(전체 평균 {baseline:.1%}, 변별력 {s['tag']} ×{s['lift']:.1f})")
                else:
                    lines.append(f"    - {s['label']}")
        else:
            lines.append("  · 자금세탁 모티프: 뚜렷한 모티프 없음")
        return "\n".join(lines)

    def _format_q1(self, d: dict) -> str:
        if not d:
            return "  (조회 결과 없음)"
        lt = [x for x in (d.get("lt_o") or []) if x]
        lines = [
            f"  · 송금 {self._nz(d.get('out_tx')):,}건 → {self._nz(d.get('out_acc')):,}개 계좌 (총 ₩{self._won(d.get('out_amt'))})",
            f"  · 수취 {self._nz(d.get('in_tx')):,}건 ← {self._nz(d.get('in_acc')):,}개 계좌 (총 ₩{self._won(d.get('in_amt'))})",
            f"  · 세탁 거래: 송금측 {self._nz(d.get('out_l')):,}건 / 수취측 {self._nz(d.get('in_l')):,}건",
        ]
        if lt:
            lines.append(f"  · 연루 세탁 유형: {', '.join(lt)}")
        return "\n".join(lines)

    def _format_q2(self, d: dict) -> str:
        n = self._nz(d.get("l_neighbors"), 0)
        if n == 0:
            return "  · 직접 연결된 세탁 확정 계좌 없음"
        ids = [x for x in (d.get("ids") or []) if x]
        s = f"  · 직접 연결된 세탁 확정 계좌 {n:,}개"
        if ids:
            s += f" (예: {', '.join(str(i) for i in ids)})"
        return s

    def _format_q3(self, paths: list[dict]) -> str:
        if not paths:
            return "  · 3단계 이내 세탁 계좌로 이어지는 자금 경로 없음"
        out = []
        for i, p in enumerate(paths, 1):
            accs = " → ".join(str(a) for a in (p.get("accs") or []))
            out.append(f"  경로 {i} ({self._nz(p.get('plen'),'?')}단계) | 종착 세탁계좌 {p.get('target','?')} "
                       f"(위험도 {self._nz(p.get('prob'),0.0):.0%})\n    └─ {accs}")
        return "\n".join(out)

    def _format_q4(self, d: dict) -> str:
        if not d:
            return "  (조회 결과 없음)"
        fo = self._nz(d.get("fo"), 0); fi = self._nz(d.get("fi"), 0)
        afo = self._nz(d.get("avg_fo"), 0.0); afi = self._nz(d.get("avg_fi"), 0.0)
        lines = [
            f"  · 송금 대상 계좌 수 {fo:,}개 (전체 평균 {afo:.1f}개)",
            f"  · 수취 출처 계좌 수 {fi:,}개 (전체 평균 {afi:.1f}개)",
        ]
        if afo > 0 and fo >= afo * 3:
            lines.append("  · [주의] 평균 대비 3배 이상 분산 송금 — 팬아웃(자금 분산) 허브 의심")
        if afi > 0 and fi >= afi * 3:
            lines.append("  · [주의] 평균 대비 3배 이상 자금 집결 — 팬인(스머핑) 허브 의심")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 핵심 의심 근거 요약 + 우선순위별 권고 조치 (결정적 생성 — 보고서 항상 포함)
    # ------------------------------------------------------------------
    def _key_grounds(self, q0, q1, q2, q4, sig_eval) -> list:
        """데이터에서 검증 가능한 구체적 의심 근거를 강한 순으로 추린다."""
        g = []
        in_l = self._nz(q1.get("in_l"), 0); out_l = self._nz(q1.get("out_l"), 0)
        if in_l or out_l:
            seg = []
            if in_l:
                seg.append(f"수취측 {in_l}건")
            if out_l:
                seg.append(f"송금측 {out_l}건")
            g.append(f"자금세탁 확정 거래 직접 연루 ({', '.join(seg)})")
        ln = self._nz(q2.get("l_neighbors"), 0)
        if ln:
            ids = [x for x in (q2.get("ids") or []) if x]
            g.append(f"자금세탁 확정 계좌 {ln}개와 직접 거래"
                     + (f" (예: {', '.join(str(i) for i in ids[:3])})" if ids else ""))
        fo = self._nz(q4.get("fo"), 0); afo = self._nz(q4.get("avg_fo"), 0.0)
        if afo > 0 and fo >= afo * 3:
            g.append(f"{fo}개 계좌로 분산 송금 — 전체 평균({afo:.1f}개)의 {fo/afo:.1f}배 (팬아웃 허브)")
        fi = self._nz(q4.get("fi"), 0); afi = self._nz(q4.get("avg_fi"), 0.0)
        if afi > 0 and fi >= afi * 3:
            g.append(f"{fi}개 계좌로부터 자금 집결 — 전체 평균({afi:.1f}개)의 {fi/afi:.1f}배 (팬인/스머핑)")
        # 변별력 높은 모티프 1개 보강
        for s in sig_eval:
            if s.get("lift") and s["lift"] >= 1.3:
                g.append(f"{s['label']} — 해당 신호 보유 계좌 세탁율 {s['rate']:.0%} (평균 대비 ×{s['lift']:.1f})")
                break
        if not g:
            prob = self._nz(q0.get("fraud_prob"), 0.0)
            g.append(f"모델 의심도 {prob:.1%} (네트워크 학습 기반 자동 선별 — 검증 가능한 행동 근거는 미약)")
        return g[:4]

    def _recommend_actions(self, tier, acc, out_l, in_l, q2, q4) -> list:
        """수사 우선순위(tier)와 실제 근거를 반영한 구체적 권고 조치."""
        ids = [x for x in (q2.get("ids") or []) if x]
        id_str = (", ".join(str(i) for i in ids[:3])) if ids else ""
        fo = self._nz(q4.get("fo"), 0); afo = self._nz(q4.get("avg_fo"), 0.0)
        is_hub = afo > 0 and fo >= afo * 3

        if tier == "confirmed":
            a = [f"해당 계좌({acc})의 출금·이체 거래를 일시 제한하고 자금 동결 여부를 즉시 검토한다."]
            seg = (f"수취 {in_l}건" if in_l else "") + (" 및 " if (in_l and out_l) else "") + (f"송금 {out_l}건" if out_l else "")
            a.append(f"자금세탁 확정 거래({seg})의 원본 거래내역·증빙을 확보하고 자금 출처와 사용처를 추적한다.")
            if id_str:
                a.append(f"직접 연결된 자금세탁 확정 계좌({id_str})를 동반 조사 대상에 포함한다.")
            if is_hub:
                a.append(f"{fo}개 분산 송금 수취 계좌의 후속 자금 이동(2차 분산 여부)을 추적한다.")
            a.append("특정금융정보법 제4조에 따라 금융정보분석원(KoFIU)에 의심거래보고(STR)를 즉시 제출한다.")
        elif tier == "network":
            a = [f"해당 계좌({acc})의 거래 모니터링을 강화하고 추가 이체 발생 시 사전 점검한다."]
            if id_str:
                a.append(f"연결된 자금세탁 확정 계좌({id_str})와의 거래 내역을 정밀 대조하고 공모 여부를 검토한다.")
            if is_hub:
                a.append(f"{fo}개 계좌로의 분산 송금 경로와 수취 계좌 위험도를 추적한다.")
            a.append("거래내역 정밀 분석 후 혐의가 보강되면 의심거래보고(STR) 제출을 검토한다.")
        elif tier == "motif":
            a = [f"해당 계좌({acc})의 거래내역을 정밀 분석하여 팬인/팬아웃·경유 패턴의 자금 흐름을 확인한다."]
            a.append("주요 거래 상대 계좌의 위험도와 거래 목적의 실재 여부를 확인한다.")
            a.append("구체적 혐의 근거 확보 시 의심거래보고(STR) 제출을 검토한다.")
        elif tier == "review":
            a = [f"해당 계좌({acc})의 원본 거래내역을 인적 검토하여 모델 의심도의 실질 근거를 확인한다."]
            a.append("일정 기간(예: 30일) 거래 모니터링을 강화하고 패턴 변화를 관찰한다.")
            a.append("변별력 있는 행동·네트워크 근거가 확인되면 의심거래보고(STR)를 검토한다.")
        elif tier == "monitor":
            a = [f"해당 계좌({acc})를 상시 모니터링 대상으로 등록하고 임계 초과 거래를 점검한다.",
                 "다음 평가 주기에 위험도 변화를 재확인한다."]
        else:
            a = [f"해당 계좌({acc})는 정기 점검 대상으로 유지한다."]
        return a

    # ------------------------------------------------------------------
    def format_context(self, node_idx: int, account_id: str = "") -> str:
        if not self._ensure_connected():
            return ""
        try:
            q0 = self._q0_profile(node_idx)
            q1 = self._q1_counterparties(node_idx)
            q2 = self._q2_launder_neighbors(node_idx)
            q3 = self._q3_path_to_launder(node_idx)
            q4 = self._q4_fanout(node_idx)
        except Exception as e:
            warnings.warn(f"[GraphRAG] 쿼리 오류: {e}", RuntimeWarning)
            return ""

        baseline, sig_eval, disc = self._signal_assessment(q0)
        model_prob = self._nz(q0.get("fraud_prob"), 0.0)
        out_l = self._nz(q1.get("out_l"), 0)
        in_l = self._nz(q1.get("in_l"), 0)
        own_l = out_l + in_l
        l_neighbors = self._nz(q2.get("l_neighbors"), 0)
        net_corro = (l_neighbors > 0) or bool(q3) or (own_l > 0)

        if (own_l > 0) and model_prob >= 0.7:
            tier, priority = "confirmed", "고위험 — 본 계좌 거래에서 자금세탁 확정 거래 직접 확인"
            judge = "본 계좌가 자금세탁으로 확정된 거래에 직접 연루되어 있어 즉시 조사가 필요함."
        elif net_corro and model_prob >= 0.7:
            tier, priority = "network", "고위험 — 자금세탁 계좌와의 네트워크 연계 확인"
            judge = "자금세탁 확정 계좌와 직접 연결되어 있어(또는 자금 경로 존재) 우선 조사가 필요함."
        elif disc >= 2 and model_prob >= 0.7:
            tier, priority = "motif", "중~고위험 — 자금세탁 모티프 복수 확인"
            judge = "기저율 대비 변별력 있는 자금세탁 모티프(팬인/팬아웃·경유 등)가 복수 확인됨."
        elif model_prob >= 0.7:
            tier, priority = "review", "검토 필요 (1차 의심 후보)"
            judge = ("모델 의심도는 높으나 검증 가능한 모티프·네트워크 근거가 충분치 않아 "
                     "확정 전 거래내역에 대한 인적 검토가 필요함.")
        elif model_prob >= 0.4:
            tier, priority = "monitor", "중위험 — 상시 모니터링"
            judge = "단정적 위험 징후는 없으나 모델 의심도가 중간 수준으로 지속 관찰이 필요함."
        else:
            tier, priority = "low", "저위험"
            judge = "특이 위험 징후 없음."

        acc = account_id or (q0.get("account_id") if q0 else "") or ""
        grounds = self._key_grounds(q0, q1, q2, q4, sig_eval)
        actions = self._recommend_actions(tier, acc, out_l, in_l, q2, q4)

        sections = [
            f"[거래 네트워크 분석 — 분석 대상 계좌 {node_idx}" + (f" ({acc})" if acc else "") + "]",
            "",
            "▶ 종합 수사 평가 (Investigative Assessment)",
            f"  · 수사 우선순위 권고     : {priority}",
            f"  · 검증 가능한 모티프 근거: 기저율 대비 변별력 있는 신호 {disc}개",
            f"  · 네트워크 연계 근거     : {'있음' if net_corro else '없음'}",
            f"  · 판단                  : {judge}",
            "  · 핵심 의심 근거(요약)   :",
            *[f"      {i}. {g}" for i, g in enumerate(grounds, 1)],
            "  · 권고 조치(우선순위 반영):",
            *[f"      {i}. {a}" for i, a in enumerate(actions, 1)],
            "",
            "▶ 1. 계좌 위험 프로파일 (거래 행태·모티프)",
            self._format_q0(q0, sig_eval, baseline),
            "",
            "▶ 2. 직접 거래 요약",
            self._format_q1(q1),
            "",
            "▶ 3. 직접 연결 세탁 계좌",
            self._format_q2(q2),
            "",
            "▶ 4. 세탁 계좌로의 자금 흐름 경로 (3단계 이내)",
            self._format_q3(q3),
            "",
            "▶ 5. 분산/집중 지표 (팬아웃·팬인)",
            self._format_q4(q4),
        ]
        return "\n".join(sections)


# 편의 함수 — 싱글톤
_retriever_instance: GraphRAGRetriever | None = None


def get_graph_retriever(lazy: bool = False) -> GraphRAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        _retriever_instance = GraphRAGRetriever(
            uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, lazy=lazy
        )
    return _retriever_instance


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    r = GraphRAGRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    if not r.is_available:
        print("Neo4j 연결 불가"); sys.exit(1)
    with r._driver.session() as s:
        row = s.run("MATCH (a:Account) WHERE a.is_laundering=true "
                    "RETURN a.node_idx AS i, a.account_id AS a ORDER BY a.fraud_prob DESC LIMIT 1").single()
    if row:
        print(r.format_context(row["i"], row["a"]))
    r.close()
