from __future__ import annotations

import warnings
from typing import Any

from neo4j import GraphDatabase


# GraphRAGRetriever

class GraphRAGRetriever: # Neo4j 그래프 쿼리 기반 GraphRAG 컨텍스트 생성기
    def __init__(
        self,
        uri: str      = "bolt://localhost:7687",
        user: str     = "neo4j",
        password: str = "qwer1234",
        max_paths: int = 5,
        lazy: bool    = False,  # True 면 첫 쿼리 시점까지 연결을 미룸 (앱 기동 단축)
    ) -> None:
        self._uri      = uri
        self._user     = user
        self._password = password
        self._max_paths = max_paths
        self._driver   = None
        self._available = False
        self._connect_attempted = False
        self._pop_stats = None  # 기저율(base-rate) 통계 캐시 — 신호 변별력 계산용

        # AuraDB(neo4j+s://) TLS 핸드셰이크는 최대 30초 — lazy 모드에서는
        # 앱 기동을 막지 않도록 첫 format_context() 호출 시점에 연결한다.
        if not lazy:
            self._ensure_connected()

    # ------------------------------------------------------------------
    # 연결 관리
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        driver = None
        try:
            driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=5,
                # AuraDB(neo4j+s://)는 TLS 핸드셰이크로 초기 연결이 느림
                # → 로컬 bolt://는 5초면 충분하지만 클라우드는 30초 필요
                connection_acquisition_timeout=30,
                liveness_check_timeout=2,
            )
            driver.verify_connectivity()
        except Exception as e:
            # 드라이버 생성 후 verify 실패 시 반드시 닫아서 연결 풀 누수 방지
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass
            self._available = False
            warnings.warn(
                f"[GraphRAG] Neo4j 연결 실패 — 그래프 RAG 비활성화: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        # 연결 상태 확정은 try 블록 밖에서 — 로깅 등 부수 작업의 실패가
        # 성공한 연결을 실패로 둔갑시키지 않도록 분리한다.
        # (과거 버그: cp949 콘솔에서 이모지 print가 UnicodeEncodeError를 던져
        #  except 블록이 멀쩡한 드라이버를 닫고 _available=False 로 만들었음)
        self._driver    = driver
        self._available = True
        try:
            print("[GraphRAG] Neo4j 연결 성공")
        except Exception:
            pass

    def _ensure_connected(self) -> bool:
        """연결을 1회만 시도하고 결과를 캐시합니다 (lazy 모드 지원)."""
        if not self._connect_attempted:
            self._connect_attempted = True
            self._try_connect()
        return self._available

    @property
    def is_available(self) -> bool: # Neo4j 연결이 활성화되어 있는지 반환합니다.
        return self._available

    @property
    def connect_attempted(self) -> bool: # 연결 시도 여부 (lazy 모드 상태 표시용)
        return self._connect_attempted

    @property
    def is_configured(self) -> bool: # 접속 정보가 설정되어 있는지 (연결 시도 없이 판단)
        return bool(self._uri and self._password)

    def close(self) -> None: # 드라이버 연결 해제
        if self._driver:
            self._driver.close()

    def _run(self, query: str, **params) -> list[dict[str, Any]]: # Cypher 쿼리 실행 후 결과 반환
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]


    # Cypher 쿼리

    def _q0_account_profile(self, node_idx: int) -> dict:
        """계좌 자체의 거래 행동 프로파일 + 전체 대비 위험도 순위.

        이웃이 희소한 계좌(PaySim 단발성 거래)도 자체 속성으로 구체적
        의심 정황(계좌 비우기·잔액 불일치·수취 부재 등)을 제시하기 위함.
        """
        rows = self._run(
            """
            MATCH (a:Account {node_idx: $node_idx})
            CALL { MATCH (t:Account) RETURN count(t) AS total }
            CALL { WITH a
                   MATCH (x:Account) WHERE x.fraud_prob >= a.fraud_prob
                   RETURN count(x) AS rank_pos }
            RETURN a.account_id          AS account_id,
                   a.fraud_prob          AS fraud_prob,
                   a.is_fraud            AS is_fraud,
                   a.send_count          AS send_count,
                   a.recv_count          AS recv_count,
                   a.send_max            AS send_max_log,
                   a.zero_balance_cnt    AS zero_balance_cnt,
                   a.mismatch_sum        AS mismatch_sum,
                   a.empty_acct_recv     AS empty_acct_recv,
                   total,
                   rank_pos
            """,
            node_idx=node_idx,
        )
        return rows[0] if rows else {}

    def _population_stats(self) -> dict:
        """전체 계좌 대비 각 행동 신호의 기저율·사기율 (1회 계산 후 캐시).

        신호의 '변별력(lift)' = (신호 보유 계좌 사기율) / (전체 사기율) 을 계산해,
        흔하기만 하고 사기를 구분 못 하는 신호를 약한 근거로 정직하게 표기하기 위함.
        """
        if self._pop_stats is not None:
            return self._pop_stats
        rows = self._run(
            """
            MATCH (a:Account)
            RETURN count(a) AS total,
                   sum(CASE WHEN a.is_fraud THEN 1 ELSE 0 END) AS fraud_total,
                   sum(CASE WHEN a.zero_balance_cnt >= 1 THEN 1 ELSE 0 END) AS zb_t,
                   sum(CASE WHEN a.zero_balance_cnt >= 1 AND a.is_fraud THEN 1 ELSE 0 END) AS zb_f,
                   sum(CASE WHEN a.mismatch_sum >= 1 THEN 1 ELSE 0 END) AS ms_t,
                   sum(CASE WHEN a.mismatch_sum >= 1 AND a.is_fraud THEN 1 ELSE 0 END) AS ms_f,
                   sum(CASE WHEN a.recv_count = 0 AND a.send_count >= 1 THEN 1 ELSE 0 END) AS out_t,
                   sum(CASE WHEN a.recv_count = 0 AND a.send_count >= 1 AND a.is_fraud THEN 1 ELSE 0 END) AS out_f,
                   sum(CASE WHEN a.empty_acct_recv >= 1 THEN 1 ELSE 0 END) AS er_t,
                   sum(CASE WHEN a.empty_acct_recv >= 1 AND a.is_fraud THEN 1 ELSE 0 END) AS er_f
            """
        )
        self._pop_stats = rows[0] if rows else {}
        return self._pop_stats

    def _signal_assessment(self, q0: dict) -> tuple:
        """노드가 보유한 행동 신호별 변별력(lift) 평가.

        Returns (baseline_fraud_rate, [{label, rate, lift, tag}...], discriminative_count)
        discriminative_count = lift >= 1.3 인 신호 수 (실제 의심 근거가 되는 신호).
        """
        pop = self._population_stats()
        total = self._nz(pop.get("total"), 0)
        baseline = (self._nz(pop.get("fraud_total"), 0) / total) if total else 0.0

        sc = int(self._nz(q0.get("send_count"), 0))
        rc = int(self._nz(q0.get("recv_count"), 0))
        zb = int(self._nz(q0.get("zero_balance_cnt"), 0))
        ms = int(self._nz(q0.get("mismatch_sum"), 0))
        er = int(self._nz(q0.get("empty_acct_recv"), 0))

        candidates = []
        if zb > 0:
            candidates.append(("송금 직후 잔액 0원(계좌 비우기)", "zb"))
        if rc == 0 and sc > 0:
            candidates.append(("수취 없이 송금만 발생(자금 유출 전용 가능성)", "out"))
        if ms > 0:
            candidates.append(("잔액 불일치", "ms"))
        if er > 0:
            candidates.append(("빈 계좌로 수취(자금 경유 가능성)", "er"))

        evaluated = []
        disc = 0
        for label, key in candidates:
            t = self._nz(pop.get(f"{key}_t"), 0)
            f = self._nz(pop.get(f"{key}_f"), 0)
            rate = (f / t) if t else None
            lift = (rate / baseline) if (rate is not None and baseline) else None
            if lift is not None and lift >= 1.3:
                disc += 1
            tag = (
                "높음" if (lift and lift >= 2.0)
                else "보통" if (lift and lift >= 1.3)
                else "낮음"
            )
            evaluated.append({"label": label, "rate": rate, "lift": lift, "tag": tag})
        return baseline, evaluated, disc

    def _q1_direct_connections(self, node_idx: int) -> dict: # 직접 연결 계좌 요약(1-hop)
        rows = self._run(
            """
            MATCH (center:Account {node_idx: $node_idx})

            // 송금 방향 (center → 수취 계좌)
            OPTIONAL MATCH (center)-[r_out:SENT_TO]->(out:Account)
            WITH center,
                 count(DISTINCT out)               AS out_count,
                 sum(r_out.amount)                 AS total_sent,
                 sum(CASE WHEN r_out.is_fraud_tx THEN 1 ELSE 0 END) AS fraud_tx_out,
                 collect(DISTINCT CASE WHEN out.is_fraud THEN out.account_id END)[..5]
                                                   AS fraud_out_ids

            // 수취 방향 (송금 계좌 → center)
            OPTIONAL MATCH (in_acc:Account)-[r_in:SENT_TO]->(center)
            RETURN center.account_id               AS account_id,
                   out_count,
                   count(DISTINCT in_acc)          AS in_count,
                   total_sent,
                   sum(r_in.amount)                AS total_received,
                   fraud_tx_out,
                   sum(CASE WHEN r_in.is_fraud_tx THEN 1 ELSE 0 END) AS fraud_tx_in,
                   fraud_out_ids
            """,
            node_idx=node_idx,
        )
        return rows[0] if rows else {}

    def _q2_fraud_cluster(self, node_idx: int) -> dict: # 사기 클러스터 분석(2-hop)
        rows = self._run(
            """
            MATCH (center:Account {node_idx: $node_idx})
            MATCH (center)-[:SENT_TO*1..2]-(neighbor:Account)
            WHERE neighbor.node_idx <> $node_idx

            WITH collect(DISTINCT neighbor) AS cluster
            UNWIND cluster AS node
            RETURN count(node)                                              AS cluster_size,
                   sum(CASE WHEN node.is_fraud     THEN 1 ELSE 0 END)      AS fraud_count,
                   sum(CASE WHEN node.fraud_prob > 0.7 THEN 1 ELSE 0 END)  AS high_risk_count,
                   round(avg(node.fraud_prob) * 10000) / 10000.0            AS avg_fraud_prob,
                   max(node.fraud_prob)                                     AS max_fraud_prob,
                   collect(CASE WHEN node.is_fraud
                           THEN node.account_id END)[..5]                  AS fraud_account_ids
            """,
            node_idx=node_idx,
        )
        return rows[0] if rows else {}

    def _q3_fraud_paths(self, node_idx: int) -> list[dict]: # 자금 흐름 경로 탐지(3-hop 이내 사기 계좌까지의 경로)
        rows = self._run(
            f"""
            MATCH (center:Account {{node_idx: $node_idx}})
            MATCH path = shortestPath(
                (center)-[:SENT_TO*1..3]->(fraud:Account)
            )
            WHERE fraud.is_fraud = true
                AND fraud.node_idx <> $node_idx
            WITH path,
                fraud,
                length(path)   AS path_len,
                [n IN nodes(path) | n.account_id] AS path_accounts
            WITH path_len,
                fraud.account_id      AS fraud_account,
                 round(fraud.fraud_prob * 100) / 100.0 AS fraud_prob,
                path_accounts,
                [r IN relationships(path) | r.amount] AS amounts
            RETURN path_len,
                    fraud_account,
                    fraud_prob,
                    path_accounts,
                    reduce(s = 0.0, a IN amounts | s + a) AS path_total_amount
            ORDER BY path_len ASC, fraud_prob DESC
            LIMIT {self._max_paths}
            """,
            node_idx=node_idx,
        )
        return rows

    def _q4_hub_indicator(self, node_idx: int) -> dict: # 허브 의심 지표
        rows = self._run(
            """
            // ── 핵심 수정: CALL {} 서브쿼리로 전체 그래프 통계를 분리 계산 ──
            // 기존 방식(MATCH all_nodes + OPTIONAL MATCH)은 모든 노드×관계를
            // 메모리에 올려 Neo4j JVM OOM을 유발했음 → count만 집계하도록 변경

            // Step1: 전체 통계 (각각 독립 집계 — 교차 곱 없음)
            CALL {
                MATCH (n:Account) RETURN count(n) AS total_nodes
            }
            CALL {
                MATCH ()-[:SENT_TO]->() RETURN count(*) AS total_edges
            }

            // Step2: 대상 계좌 통계
            MATCH (center:Account {node_idx: $node_idx})
            OPTIONAL MATCH (center)-[r:SENT_TO]->(out:Account)
            WITH total_edges, total_nodes, center,
                count(r)                                              AS out_degree,
                count(DISTINCT out)                                   AS unique_receivers,
                sum(CASE WHEN out.fraud_prob > 0.7 THEN 1 ELSE 0 END) AS high_risk_receivers

            // Step3: 배율 계산
            WITH out_degree, unique_receivers, high_risk_receivers,
                 CASE WHEN total_nodes > 0
                     THEN round(toFloat(total_edges) / total_nodes * 100) / 100.0
                     ELSE 0.0 END AS avg_out_degree,
                 CASE WHEN total_nodes > 0 AND total_edges > 0
                     THEN round(
                             toFloat(out_degree) /
                             (toFloat(total_edges) / total_nodes) * 100
                          ) / 100.0
                     ELSE 0.0 END AS degree_ratio

            RETURN out_degree,
                   unique_receivers,
                   high_risk_receivers,
                   avg_out_degree,
                   degree_ratio
            """,
            node_idx=node_idx,
        )
        return rows[0] if rows else {}


    # 컨텍스트 포맷터

    @staticmethod
    def _fmt_amount(amount) -> str: # 금액을 읽기 편한 문자열로 변환
        if amount is None:
            return "0"
        a = float(amount)
        if a >= 1_000_000:
            return f"{a/1_000_000:.2f}M"
        if a >= 1_000:
            return f"{a:,.0f}"
        return f"{a:.2f}"

    @staticmethod
    def _nz(value, default=0):
        """Cypher 집계가 null 을 반환하는 경우 기본값으로 치환.

        avg()/max() 는 입력이 0행이면 null 을 반환하므로 (관계 미적재 등)
        숫자 포맷(:.2% 등) 적용 전 반드시 None 을 걸러야 한다.
        """
        return default if value is None else value

    def _format_q0(self, d: dict, sig_eval: list, baseline: float) -> str:
        """계좌 자체 위험 프로파일 — 각 행동 신호를 기저율 대비 변별력과 함께 표기.

        흔하기만 한 신호(변별력 낮음)를 강한 근거처럼 포장하지 않는다.
        """
        import math
        if not d:
            return "  (계좌 정보 없음)"
        prob  = self._nz(d.get("fraud_prob"), 0.0)
        total = self._nz(d.get("total"), 0)
        rank  = self._nz(d.get("rank_pos"), 0)
        pct   = (rank / total * 100) if total else 0.0
        sc    = int(self._nz(d.get("send_count"), 0))
        rc    = int(self._nz(d.get("recv_count"), 0))
        smax  = d.get("send_max_log")
        smax_won = math.expm1(float(smax)) if smax else 0.0

        lines = [
            f"  · 모델 의심도(임베딩 기반): {prob:.2%}  (전체 {total:,}개 계좌 중 상위 {pct:.2f}%)",
            "    └─ 해석: GraphSAGE 가 이 계좌의 '거래 행태 + 주변 자금 흐름 구조'(거래 규모·입출금",
            "       방향·거래 상대 구성·다단계 자금 경로의 형태)를 수치 지문(임베딩)으로 요약한 결과,",
            "       과거 사기로 확인된 계좌들의 지문과 통계적으로 매우 가깝다는 의미입니다. 즉 '알려진",
            "       자금세탁 계좌와 거래 행태·구조가 유사'하다는 유형 기반 선별 신호이며, 특정 거래의",
            "       위법성을 입증하는 직접 증거는 아닙니다 (확정은 원본 거래내역 인적 검토 필요).",
            f"  · 거래 활동             : 송금 {sc:,}건 / 수취 {rc:,}건",
            f"  · 최대 단일 송금액      : 약 ₩{self._fmt_amount(smax_won)}",
        ]
        if sig_eval:
            lines.append("  · 행동 신호 (기저율 대비 변별력):")
            for s in sig_eval:
                if s["lift"] is not None:
                    lines.append(
                        f"    - {s['label']}: 해당 신호 보유 계좌 사기율 {s['rate']:.1%} "
                        f"(전체 평균 {baseline:.1%}, 변별력 {s['tag']} ×{s['lift']:.2f})"
                    )
                else:
                    lines.append(f"    - {s['label']}")
        return "\n".join(lines)

    def _format_q1(self, data: dict) -> str:
        if not data:
            return "  (조회 결과 없음)"
        lines = [
            f"  · 계좌 ID            : {data.get('account_id', '알 수 없음')}",
            f"  · 송금 대상 계좌 수  : {self._nz(data.get('out_count'), 0):,}개",
            f"  · 수취 발신 계좌 수  : {self._nz(data.get('in_count'), 0):,}개",
            f"  · 총 송금액          : ₩{self._fmt_amount(data.get('total_sent'))}",
            f"  · 총 수취액          : ₩{self._fmt_amount(data.get('total_received'))}",
            f"  · 사기 거래 (송금)   : {self._nz(data.get('fraud_tx_out'), 0):,}건",
            f"  · 사기 거래 (수취)   : {self._nz(data.get('fraud_tx_in'), 0):,}건",
        ]
        fraud_ids = [x for x in (data.get("fraud_out_ids") or []) if x]
        if fraud_ids:
            lines.append(f"  · 직접 연결 사기 계좌: {', '.join(fraud_ids)}")
        return "\n".join(lines)

    def _format_q2(self, data: dict) -> str:
        if not data:
            return "  (조회 결과 없음)"
        # 클러스터가 비면 avg()/max() 가 null → None 안전 처리 필수
        cluster_size  = self._nz(data.get("cluster_size"), 0)
        fraud_count   = self._nz(data.get("fraud_count"), 0)
        high_risk     = self._nz(data.get("high_risk_count"), 0)
        avg_prob      = self._nz(data.get("avg_fraud_prob"), 0.0)
        max_prob      = self._nz(data.get("max_fraud_prob"), 0.0)
        fraud_ratio   = (fraud_count / cluster_size * 100) if cluster_size else 0
        fraud_ids     = [x for x in (data.get("fraud_account_ids") or []) if x]

        if cluster_size == 0:
            return ("  · 2단계 이내 연결된 다른 계좌 없음\n"
                    "    └─ 단발성 거래 계좌의 전형적 특징 (자금을 한 번에 인출 후 미활동). "
                    "넓은 거래망이 없다는 점 자체가 '치고 빠지는' 패턴과 부합합니다.")

        lines = [
            f"  · 2-hop 클러스터 크기   : {cluster_size:,}개 계좌",
            f"  · 사기 확정 계좌 수     : {fraud_count:,}개 ({fraud_ratio:.1f}%)",
            f"  · 고위험(>70%) 계좌 수  : {high_risk:,}개",
            f"  · 클러스터 평균 위험도  : {avg_prob:.2%}",
            f"  · 클러스터 최고 위험도  : {max_prob:.2%}",
        ]
        if fraud_ids:
            lines.append(f"  · 주요 사기 계좌        : {', '.join(fraud_ids)}")
        return "\n".join(lines)

    def _format_q3(self, paths: list[dict]) -> str:
        if not paths:
            return ("  · 3단계 이내에서 알려진 사기 계좌로 이어지는 직접 경로 없음\n"
                    "    └─ 직접 연결된 확정 사기 계좌는 없으나, 위 '계좌 위험 프로파일'의 "
                    "행동 신호와 모델 위험도를 근거로 의심함.")
        lines = []
        for i, p in enumerate(paths, 1):
            accs   = p.get("path_accounts") or []
            arrow  = " → ".join(str(a) for a in accs)
            amount = self._fmt_amount(p.get("path_total_amount"))
            prob   = self._nz(p.get("fraud_prob"), 0.0)
            lines.append(
                f"  경로 {i} ({p.get('path_len', '?')}hop) | "
                f"최종 사기 계좌: {p.get('fraud_account', '?')} "
                f"(위험도 {prob:.0%}) | 이동금액: ₩{amount}\n"
                f"    └─ {arrow}"
            )
        return "\n".join(lines)

    def _format_q4(self, data: dict) -> str:
        if not data:
            return "  (조회 결과 없음)"
        out_deg   = self._nz(data.get("out_degree"), 0)
        uniq_recv = self._nz(data.get("unique_receivers"), 0)
        hi_recv   = self._nz(data.get("high_risk_receivers"), 0)
        avg_deg   = self._nz(data.get("avg_out_degree"), 1.0)
        ratio     = self._nz(data.get("degree_ratio"), 0.0)
        hi_ratio  = (hi_recv / uniq_recv * 100) if uniq_recv else 0

        lines = [
            f"  · 총 송금 건수              : {out_deg:,}건",
            f"  · 고유 수취 계좌 수         : {uniq_recv:,}개",
            f"  · 고위험 수취 계좌 수       : {hi_recv:,}개 ({hi_ratio:.1f}%)",
            f"  · 전체 평균 대비 송금 배율  : {ratio:.1f}배 (전체 평균 {avg_deg:.1f}건)",
        ]
        if ratio >= 3.0:
            lines.append("  · [주의] 허브 계좌 의심: 평균의 3배 이상 송금")
        return "\n".join(lines)


    # 공개 메서드

    def query_all(self, node_idx: int) -> dict[str, Any]: # 4개 쿼리 실행 후 원본 결과 딕셔너리에 반환
        return {
            "q1_direct" : self._q1_direct_connections(node_idx),
            "q2_cluster": self._q2_fraud_cluster(node_idx),
            "q3_paths"  : self._q3_fraud_paths(node_idx),
            "q4_hub"    : self._q4_hub_indicator(node_idx),
        }

    def format_context(
        self,
        node_idx: int,
        account_id: str = "",
    ) -> str:
        # lazy 모드: 첫 호출 시점에 연결 (실패 시 빈 컨텍스트로 우아하게 강등)
        if not self._ensure_connected():
            return ""

        try:
            q0 = self._q0_account_profile(node_idx)
            q1 = self._q1_direct_connections(node_idx)
            q2 = self._q2_fraud_cluster(node_idx)
            q3 = self._q3_fraud_paths(node_idx)
            q4 = self._q4_hub_indicator(node_idx)
        except Exception as e:
            warnings.warn(f"[GraphRAG] 쿼리 실행 오류: {e}", RuntimeWarning)
            return ""

        # account_id 가 비면 Q0 조회 결과에서 보강
        acc = account_id or (q0.get("account_id") if q0 else "") or ""
        header = (
            f"[거래 네트워크 분석 — 분석 대상 계좌 {node_idx}"
            + (f" ({acc})" if acc else "")
            + "]"
        )

        # ── 종합 수사 평가 (증거 강도 기반 우선순위 보정) ──────────────────
        baseline, sig_eval, disc = self._signal_assessment(q0)
        model_prob = self._nz(q0.get("fraud_prob"), 0.0)
        q2_fraud   = self._nz(q2.get("fraud_count"), 0)
        q2_hr      = self._nz(q2.get("high_risk_count"), 0)
        net_corro  = (q2_fraud > 0) or (q2_hr > 0) or bool(q3)

        if net_corro and model_prob >= 0.7:
            priority = "고위험 — 네트워크 연계 근거 확인"
            judgement = "거래 그래프상 사기 계좌와의 직접 연계가 확인됨. 우선 조사 권고."
        elif disc >= 2 and model_prob >= 0.7:
            priority = "중~고위험 — 행동 근거 확인"
            judgement = "기저율 대비 변별력 있는 거래 행동 신호가 복수 확인됨. 거래내역 정밀 검토 권고."
        elif model_prob >= 0.7:
            priority = "검토 필요 (1차 의심 후보)"
            judgement = (
                "모델 의심도는 높으나, 이는 거래 그래프 임베딩 유사성에 기반한 자동 선별 점수임. "
                "기저율 대비 변별력 있는 행동 근거와 직접적 네트워크 연계가 확인되지 않아, "
                "확정 전 원본 거래내역에 대한 인적 검토가 필요함."
            )
        elif model_prob >= 0.4:
            priority = "중위험 — 상시 모니터링"
            judgement = "단정적 위험 징후는 없으나 모델 의심도가 중간 수준임. 모니터링 권고."
        else:
            priority = "저위험"
            judgement = "특이 위험 징후 없음."

        assessment = [
            "▶ 종합 수사 평가 (Investigative Assessment)",
            f"  · 수사 우선순위 권고     : {priority}",
            f"  · 검증 가능한 행동 근거  : 기저율 대비 변별력 있는 신호 {disc}개",
            f"  · 네트워크 연계 근거     : {'있음' if net_corro else '없음'}",
            f"  · 판단                  : {judgement}",
        ]

        sections = [
            header,
            "",
            *assessment,
            "",
            "▶ 1. 계좌 위험 프로파일 (자체 거래 행동)",
            self._format_q0(q0, sig_eval, baseline),
            "",
            "▶ 2. 직접 거래 상대 계좌",
            self._format_q1(q1),
            "",
            "▶ 3. 인근 거래망(2단계 이내) 위험도",
            self._format_q2(q2),
            "",
            "▶ 4. 사기 계좌로 이어지는 자금 흐름 경로 (3단계 이내)",
            self._format_q3(q3),
            "",
            "▶ 5. 송금 집중도 (허브 계좌 여부)",
            self._format_q4(q4),
        ]
        return "\n".join(sections)


# 편의 함수 — Streamlit 캐시 친화적 싱글톤

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


# 직접 실행 시 연결 테스트 + 샘플 조회

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from neo4j_config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    retriever = GraphRAGRetriever(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )

    if not retriever.is_available:
        print("Neo4j에 연결할 수 없습니다. neo4j_config.py를 확인하세요.")
        sys.exit(1)

    # 고위험 계좌 1개 자동 선택
    with retriever._driver.session() as s:
        row = s.run(
            "MATCH (a:Account) WHERE a.is_fraud = true "
            "RETURN a.node_idx AS idx, a.account_id AS aid "
            "ORDER BY a.fraud_prob DESC LIMIT 1"
        ).single()

    if row:
        node_idx   = row["idx"]
        account_id = row["aid"]
        print(f"\n[테스트] 고위험 계좌: {account_id} (node_idx={node_idx})\n")
        print("=" * 60)
        context = retriever.format_context(node_idx=node_idx, account_id=account_id)
        print(context)
        print("=" * 60)
    else:
        print("데이터가 없습니다. neo4j_loader.py를 먼저 실행하세요.")

    retriever.close()
