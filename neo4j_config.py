from __future__ import annotations

import os

# ──────────────────────────────────────────────────────────────────────
# Neo4j 연결 설정
#
# 우선순위:
#   1. 환경변수 / Streamlit secrets
#   2. 아래 기본값 (로컬 개발용)
# ──────────────────────────────────────────────────────────────────────

def _get_secret(key: str, default: str) -> str:
    """환경변수 → Streamlit secrets → 기본값 순으로 설정값을 반환합니다."""
    # 환경변수 우선
    val = os.environ.get(key)
    if val:
        return val
    # Streamlit secrets (배포 환경)
    try:
        import streamlit as st  # type: ignore
        parts = key.lower().split("_", 1)   # e.g. NEO4J_URI → ["neo4j", "uri"]
        if len(parts) == 2:
            val = st.secrets.get(parts[0], {}).get(parts[1])
            if val:
                return val
    except Exception:
        pass
    return default


NEO4J_URI      = _get_secret("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = _get_secret("NEO4J_USERNAME",  "neo4j")
NEO4J_PASSWORD = _get_secret("NEO4J_PASSWORD",  "")

# LangChain Neo4j용 URL
NEO4J_URL = NEO4J_URI


# ──────────────────────────────────────────────────────────────────────
# 연결 유틸
# ──────────────────────────────────────────────────────────────────────

def get_driver():
    """
    neo4j.GraphDatabase.driver 인스턴스를 반환합니다.
    연결 실패 시 ServiceUnavailable / AuthError 등을 그대로 전파합니다.
    """
    from neo4j import GraphDatabase

    # AuraDB(neo4j+s://) 는 TLS 핸드셰이크로 초기 연결이 로컬보다 느리므로
    # connection_acquisition_timeout 을 30s 로 설정
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_pool_size=5,
        connection_acquisition_timeout=30,
        liveness_check_timeout=2,
    )
    driver.verify_connectivity()
    return driver


def test_connection() -> bool:
    """
    Neo4j 연결을 테스트하고 결과를 출력합니다.

    Returns:
        bool: 연결 성공 여부
    """
    print(f"[Neo4j] 연결 시도: {NEO4J_URI}")
    try:
        driver = get_driver()
        with driver.session() as session:
            result = session.run("RETURN 'AML GraphRAG 연결 성공' AS msg")
            msg = result.single()["msg"]
            print(f"[Neo4j] {msg}")

        with driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            print(f"[Neo4j] 현재 DB 상태 — 노드: {node_count}개 / 관계: {rel_count}개")

        driver.close()
        return True

    except Exception as e:
        err = str(e)
        print(f"[Neo4j] 연결 실패: {err}")
        print()
        _print_diagnosis(err)
        return False


def _print_diagnosis(err: str) -> None:
    """에러 메시지를 분석해 가장 가능성 높은 원인과 조치를 출력합니다."""
    err_lower = err.lower()

    if "serviceunavailable" in err_lower or "connection refused" in err_lower:
        print("▶ 원인: Neo4j가 실행되지 않았거나 Bolt 포트(7687)가 닫혀 있습니다.")
        print("  조치:")
        print("  1. Neo4j Desktop에서 인스턴스가 'Started' 상태인지 확인")
        print("  2. debug.log에서 시작 실패 원인 확인")
        print("     경로: Neo4j Desktop → Open Folder → Logs → debug.log")
        print()
        print("  ※ 'Address already in use: bind' 오류가 debug.log에 있다면:")
        print("     → Neo4j 설정 파일(neo4j.conf)에서 클러스터 포트 변경")
        print("     → Neo4j Desktop → Settings 에 아래 추가:")
        print("        initial.server.mode_constraint=NONE")
        print("     또는 클러스터 포트를 다른 번호로 변경:")
        print("        server.cluster.listen_address=localhost:6100")
        print("        server.discovery.listen_address=localhost:6101")
        print()
        print("  ※ 포트 충돌 확인 (관리자 CMD):")
        print("     netstat -ano | findstr :6000")

    elif "authentication" in err_lower or "unauthorized" in err_lower:
        print("▶ 원인: 비밀번호가 올바르지 않습니다.")
        print("  조치: neo4j_config.py의 NEO4J_PASSWORD 또는")
        print("        환경변수 NEO4J_PASSWORD 를 확인하세요.")

    elif "timeout" in err_lower:
        print("▶ 원인: 연결 시간 초과 (Neo4j 기동 중이거나 방화벽 차단)")
        print("  조치:")
        print("  1. Neo4j가 완전히 기동될 때까지 대기 후 재시도")
        print("  2. Windows 방화벽에서 7687 포트 허용 여부 확인")

    else:
        print("▶ 일반 확인 사항:")
        print("  1. Neo4j Desktop에서 인스턴스가 'Started' 상태인지 확인")
        print("  2. NEO4J_PASSWORD 환경변수 또는 neo4j_config.py 값 확인")
        print("  3. bolt://localhost:7687 포트 개방 여부 확인")


# 직접 실행 시 연결 테스