import sys
from pathlib import Path

# neo4j_config 가 임포트 시점에 환경변수를 읽으므로 가장 먼저 .env 로드.
# (없으면 NEO4J_URI 가 기본값 bolt://localhost:7687 로 폴백 → AuraDB 미접속)
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows 기본 콘솔(cp949)에서 한국어 진행 메시지 출력 시 UnicodeEncodeError 방지
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from neo4j_config import get_driver


# 스키마 DDL

SCHEMA_QUERIES = [
    # 기존 데이터 초기화 (재실행 안전)
    "MATCH (n) DETACH DELETE n",

    # UNIQUE 제약 (account_id 중복 방지 + 자동 인덱스 생성)
    """
    CREATE CONSTRAINT account_id_unique IF NOT EXISTS
    FOR (a:Account) REQUIRE a.account_id IS UNIQUE
    """,

    # 사기 라벨 인덱스 (is_fraud 필터 쿼리 최적화)
    """
    CREATE INDEX account_fraud_idx IF NOT EXISTS
    FOR (a:Account) ON (a.is_fraud)
    """,

    # 사기 확률 인덱스 (fraud_prob 범위 쿼리 최적화)
    """
    CREATE INDEX account_prob_idx IF NOT EXISTS
    FOR (a:Account) ON (a.fraud_prob)
    """,
]


def init_schema(force: bool = False) -> None: # Neo4j 스키마 초기화
    driver = get_driver()
    print("[Schema] Neo4j 스키마 초기화 시작...")

    with driver.session() as session:
        # 기존 노드/관계 수 확인
        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        if node_count > 0 and not force:
            print(
                f"[Schema] 기존 데이터 감지 ({node_count}개 노드). "
                "재초기화하려면 force=True로 실행하세요."
            )
            driver.close()
            return

        for i, query in enumerate(SCHEMA_QUERIES):
            q = query.strip()
            if not q:
                continue
            label = q[:60].replace("\n", " ")
            try:
                session.run(q)
                print(f"  [OK] [{i+1}/{len(SCHEMA_QUERIES)}] {label}...")
            except Exception as e:
                # 제약/인덱스 이미 존재 시 무시
                if "already exists" in str(e).lower():
                    print(f"  [SKIP] [{i+1}] 이미 존재 (건너뜀): {label[:40]}")
                else:
                    print(f"  [FAIL] [{i+1}] 실패: {e}")

    driver.close()
    print("[Schema] 스키마 초기화 완료.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j AML 스키마 초기화")
    parser.add_argument(
        "--force", action="store_true",
        help="기존 데이터를 삭제하고 스키마를 재초기화합니다"
    )
    args = parser.parse_args()

    init_schema(force=args.force)
