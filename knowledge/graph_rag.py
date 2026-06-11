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
            self._driver   = driver
            self._available = True
            print("[GraphRAG] Neo4j 연결 성공 ✅")
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


    # Cypher 쿼리 4종

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

    def _format_q1(self, data: dict) -> str:
        if not data:
            return "  (조회 결과 없음)"
        lines = [
            f"  · 계좌 ID            : {data.get('account_id', '알 수 없음')}",
            f"  · 송금 대상 계좌 수  : {data.get('out_count', 0):,}개",
            f"  · 수취 발신 계좌 수  : {data.get('in_count', 0):,}개",
            f"  · 총 송금액          : ₩{self._fmt_amount(data.get('total_sent'))}",
            f"  · 총 수취액          : ₩{self._fmt_amount(data.get('total_received'))}",
            f"  · 사기 거래 (송금)   : {data.get('fraud_tx_out', 0):,}건",
            f"  · 사기 거래 (수취)   : {data.get('fraud_tx_in', 0):,}건",
        ]
        fraud_ids = [x for x in (data.get("fraud_out_ids") or []) if x]
        if fraud_ids:
            lines.append(f"  · 직접 연결 사기 계좌: {', '.join(fraud_ids)}")
        return "\n".join(lines)

    def _format_q2(self, data: dict) -> str:
        if not data:
            return "  (조회 결과 없음)"
        cluster_size  = data.get("cluster_size", 0)
        fraud_count   = data.get("fraud_count", 0)
        high_risk     = data.get("high_risk_count", 0)
        avg_prob      = data.get("avg_fraud_prob", 0.0)
        max_prob      = data.get("max_fraud_prob", 0.0)
        fraud_ratio   = (fraud_count / cluster_size * 100) if cluster_size else 0
        fraud_ids     = [x for x in (data.get("fraud_account_ids") or []) if x]

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
            return "  (3-hop 이내 사기 계좌 경로 없음)"
        lines = []
        for i, p in enumerate(paths, 1):
            accs   = p.get("path_accounts") or []
            arrow  = " → ".join(str(a) for a in accs)
            amount = self._fmt_amount(p.get("path_total_amount"))
            prob   = p.get("fraud_prob", 0.0)
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
        out_deg   = data.get("out_degree", 0)
        uniq_recv = data.get("unique_receivers", 0)
        hi_recv   = data.get("high_risk_receivers", 0)
        avg_deg   = data.get("avg_out_degree", 1.0)
        ratio     = data.get("degree_ratio", 0.0)
        hi_ratio  = (hi_recv / uniq_recv * 100) if uniq_recv else 0

        lines = [
            f"  · 총 송금 건수              : {out_deg:,}건",
            f"  · 고유 수취 계좌 수         : {uniq_recv:,}개",
            f"  · 고위험 수취 계좌 수       : {hi_recv:,}개 ({hi_ratio:.1f}%)",
            f"  · 전체 평균 대비 송금 배율  : {ratio:.1f}배 (전체 평균 {avg_deg:.1f}건)",
        ]
        if ratio >= 3.0:
            lines.append("  · ⚠️  허브 계좌 의심: 평균의 3배 이상 송금")
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
            q1 = self._q1_direct_connections(node_idx)
            q2 = self._q2_fraud_cluster(node_idx)
            q3 = self._q3_fraud_paths(node_idx)
            q4 = self._q4_hub_indicator(node_idx)
        except Exception as e:
            warnings.warn(f"[GraphRAG] 쿼리 실행 오류: {e}", RuntimeWarning)
            return ""

        header = (
            f"[그래프 네트워크 분석 결과 — 노드 {node_idx}"
            + (f" / {account_id}" if account_id else "")
            + "]"
        )

        sections = [
            header,
            "",
            "▶ Q1. 직접 연결 계좌 (1-hop)",
            self._format_q1(q1),
            "",
            "▶ Q2. 사기 클러스터 (2-hop 범위)",
            self._format_q2(q2),
            "",
            "▶ Q3. 사기 계좌까지의 자금 흐름 경로 (최대 3-hop)",
            self._format_q3(q3),
            "",
            "▶ Q4. 허브 계좌 의심 지표",
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
