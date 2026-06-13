"""processed_graph_data.pt 기반 Neo4j 복구 적재 도구.

PaySim 원본 CSV(dataset/paysim1.csv)가 없을 때 사용하는 보조 적재 스크립트.
과거 적재가 중단되어 노드 일부(node_idx 72000~)와 관계 전체가 누락된 DB를
.pt 그래프 파일 + 모델 파일만으로 복구한다.

복구 범위:
  - 누락 노드: 합성 ID(ACC{idx:06d}) + is_fraud(y) + fraud_prob(XGBoost 재계산)
  - SENT_TO 관계 전체: edge_index/edge_attr 에서 복원
    (amount, log_amount, tx_type, balance_mismatch, hour_of_day_norm)

CSV 전용 정보는 복원 불가 → 생략:
  - step, is_fraud_tx (관계 속성), 원본 계좌명(누락 노드), 집계 피처
  - GraphRAG Cypher 는 is_fraud_tx 가 null 이어도 0 으로 집계되므로 동작에 지장 없음

정식 복원은 CSV 확보 후 `python tools/neo4j_loader.py --force` 를 권장한다.

사용법:
    python tools/neo4j_repair_from_pt.py            # 노드 보충 + 관계 적재
    python tools/neo4j_repair_from_pt.py --force    # 기존 SENT_TO 삭제 후 재적재
"""
from __future__ import annotations

# neo4j_config 가 임포트 시점에 환경변수를 읽으므로 가장 먼저 로드
from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Windows 기본 콘솔(cp949)에서 진행 메시지의 비-ASCII 문자 출력 시
# UnicodeEncodeError 가 나지 않도록 stdout/stderr 를 UTF-8 로 강제한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from neo4j_config import get_driver
from tools.neo4j_loader import (
    _compute_embeddings,
    _compute_fraud_probs,
    _TYPE_MAP,
    _DEFAULT_BATCH,
)

# edge_attr 컬럼 인덱스 (notebook _EDGE_FEATURE_COLS 순서)
_EA_LOG_AMOUNT   = 0
_EA_TYPE_ENCODED = 1
_EA_BAL_MISMATCH = 4
_EA_HOUR_NORM    = 5


def _ensure_node_idx_index(session) -> None:
    """관계 적재 시 MATCH 가 node_idx 를 사용하므로 인덱스 필수 (없으면 풀스캔)."""
    session.run(
        "CREATE INDEX account_node_idx IF NOT EXISTS "
        "FOR (a:Account) ON (a.node_idx)"
    )
    print("[Repair] node_idx 인덱스 확인 완료")


def _verify_alignment(session, y: torch.Tensor, probs: np.ndarray, sample: int = 500) -> None:
    """DB에 이미 있는 노드의 fraud_prob 가 .pt 기반 재계산과 일치하는지 검증.

    node_idx 정렬이 틀리면 임베딩이 다른 노드 것이 되어 fraud_prob 이 어긋난다.
    따라서 fraud_prob 일치율이 정렬의 결정적 지표다. (불일치 크면 중단)

    is_fraud 는 정렬 지표로 쓰지 않는다 — DB(neo4j_loader: 사기 거래의
    송금자/수취자 = 거래 연루 기반)와 .pt 의 y(GNN 노드 학습 라벨)는 정의가
    다른 별개 필드이므로 정렬이 맞아도 불일치한다. 참고용 경고만 출력한다.
    """
    rows = session.run(
        "MATCH (a:Account) WHERE a.node_idx IS NOT NULL "
        "RETURN a.node_idx AS idx, a.is_fraud AS f, a.fraud_prob AS p "
        f"ORDER BY rand() LIMIT {sample}"
    ).data()

    if not rows:
        print("[Repair] 기존 노드 없음 - 정렬 검증 건너뜀")
        return

    prob_rows = [r for r in rows if r["p"] is not None]
    prob_mismatch = sum(
        1 for r in prob_rows
        if abs(float(probs[int(r["idx"])]) - float(r["p"])) > 0.05
    )
    fraud_mismatch = sum(
        1 for r in rows if bool(y[int(r["idx"])].item()) != bool(r["f"])
    )
    prob_rate  = (prob_mismatch / len(prob_rows)) if prob_rows else 1.0
    fraud_rate = fraud_mismatch / len(rows)
    print(
        f"[Repair] 정렬 검증 (표본 {len(rows)}개): "
        f"fraud_prob 불일치 {prob_rate:.1%} (정렬 지표) / "
        f"is_fraud 불일치 {fraud_rate:.1%} (라벨 정의 차이, 무시)"
    )

    if not prob_rows:
        raise RuntimeError(
            "기존 노드에 fraud_prob 이 없어 정렬을 검증할 수 없습니다. "
            "수동 확인 후 진행하세요."
        )
    if prob_rate > 0.05:
        raise RuntimeError(
            ".pt 와 DB 의 node_idx 정렬이 일치하지 않습니다 "
            f"(fraud_prob 불일치 {prob_rate:.1%}). 엣지를 잘못 연결할 수 있어 "
            "중단합니다. CSV 확보 후 neo4j_loader.py --force 로 전체 재적재하세요."
        )


def _repair_missing_nodes(
    session,
    num_nodes: int,
    y: torch.Tensor,
    probs: np.ndarray,
    batch_size: int,
) -> int:
    """DB에 없는 node_idx 를 합성 ID 노드로 보충한다 (MERGE — 멱등)."""
    existing = {
        r["idx"]
        for r in session.run(
            "MATCH (a:Account) WHERE a.node_idx IS NOT NULL RETURN a.node_idx AS idx"
        )
    }
    missing = [i for i in range(num_nodes) if i not in existing]
    if not missing:
        print("[Repair] 누락 노드 없음")
        return 0

    print(f"[Repair] 누락 노드 {len(missing):,}개 보충 시작 (기존 {len(existing):,}개)")
    rows = [
        {
            "node_idx"  : int(i),
            "account_id": f"ACC{i:06d}",
            "is_fraud"  : bool(y[i].item()),
            "fraud_prob": round(float(probs[i]), 6),
        }
        for i in missing
    ]

    loaded = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        session.run(
            """
            UNWIND $rows AS r
            MERGE (a:Account {node_idx: r.node_idx})
            ON CREATE SET a.account_id = r.account_id,
                          a.is_fraud   = r.is_fraud,
                          a.fraud_prob = r.fraud_prob,
                          a.synthetic_id = true
            """,
            rows=batch,
        )
        loaded += len(batch)
        print(f"\r  노드 보충: {loaded:,}/{len(rows):,}", end="", flush=True)
    print(f"\r  노드 보충 완료: {loaded:,}개          ")
    return loaded


def _load_edges_from_pt(
    session,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch_size: int,
    force: bool,
) -> int:
    """edge_index/edge_attr 로 SENT_TO 관계를 생성한다."""
    rel_count = session.run(
        "MATCH ()-[r:SENT_TO]->() RETURN count(r) AS c"
    ).single()["c"]

    if rel_count > 0:
        if not force:
            print(
                f"[Repair] SENT_TO {rel_count:,}개가 이미 존재합니다. "
                "--force 로 삭제 후 재적재할 수 있습니다."
            )
            return 0
        print(f"[Repair] 기존 SENT_TO {rel_count:,}개 삭제 중...")
        # 대량 삭제는 배치로 (단일 트랜잭션 메모리 한도 회피)
        while True:
            deleted = session.run(
                "MATCH ()-[r:SENT_TO]->() WITH r LIMIT 10000 "
                "DELETE r RETURN count(*) AS c"
            ).single()["c"]
            if deleted == 0:
                break
        print("[Repair] 삭제 완료")

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    la  = edge_attr[:, _EA_LOG_AMOUNT].tolist()
    te  = edge_attr[:, _EA_TYPE_ENCODED].tolist()
    bm  = edge_attr[:, _EA_BAL_MISMATCH].tolist()
    hn  = edge_attr[:, _EA_HOUR_NORM].tolist()

    rows = [
        {
            "src"             : int(s),
            "dst"             : int(d),
            "amount"          : round(float(np.expm1(l)), 2),
            "log_amount"      : round(float(l), 6),
            "tx_type"         : _TYPE_MAP.get(int(round(t)), "UNKNOWN"),
            "balance_mismatch": round(float(b), 2),
            "hour_of_day_norm": round(float(h), 4),
        }
        for s, d, l, t, b, h in zip(src, dst, la, te, bm, hn)
    ]

    total  = len(rows)
    loaded = 0
    print(f"[Repair] SENT_TO 관계 {total:,}개 적재 시작...")
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        session.run(
            """
            UNWIND $rows AS r
            MATCH (src:Account {node_idx: r.src})
            MATCH (dst:Account {node_idx: r.dst})
            CREATE (src)-[:SENT_TO {
                amount          : r.amount,
                log_amount      : r.log_amount,
                tx_type         : r.tx_type,
                balance_mismatch: r.balance_mismatch,
                hour_of_day_norm: r.hour_of_day_norm
            }]->(dst)
            """,
            rows=batch,
        )
        loaded += len(batch)
        print(f"\r  관계 적재: {loaded:,}/{total:,}", end="", flush=True)
    print(f"\r  관계 적재 완료: {loaded:,}개          ")
    return loaded


def repair(force: bool = False, batch_size: int = _DEFAULT_BATCH) -> None:
    t0 = time.time()

    graph_path = ROOT / "model" / "processed_graph_data.pt"
    gnn_path   = ROOT / "gnn_model.pth"
    xgb_path   = ROOT / "fraud_model.pkl"
    for p in [graph_path, gnn_path, xgb_path]:
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    g = torch.load(str(graph_path), map_location="cpu")
    num_nodes = g["x"].shape[0]
    print(f"[Repair] 그래프: 노드 {num_nodes:,}개 / 엣지 {g['edge_index'].shape[1]:,}개")

    embs  = _compute_embeddings(g, gnn_path)
    probs = _compute_fraud_probs(embs, g, xgb_path)

    driver = get_driver()
    print("[Repair] Neo4j 연결 성공")

    with driver.session() as session:
        _ensure_node_idx_index(session)
        _verify_alignment(session, g["y"], probs)
        _repair_missing_nodes(session, num_nodes, g["y"], probs, batch_size)
        _load_edges_from_pt(session, g["edge_index"], g["edge_attr"], batch_size, force)

    # 결과 검증
    with driver.session() as session:
        n = session.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        r = session.run("MATCH ()-[x:SENT_TO]->() RETURN count(x) AS c").single()["c"]
        syn = session.run(
            "MATCH (a:Account {synthetic_id: true}) RETURN count(a) AS c"
        ).single()["c"]
    driver.close()

    print("\n" + "=" * 60)
    print(f"  복구 완료 ({time.time() - t0:.1f}초)")
    print(f"  [OK] 계좌 노드      : {n:,}개 (합성 ID: {syn:,}개)")
    print(f"  [OK] SENT_TO 관계   : {r:,}개")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="processed_graph_data.pt 로 Neo4j 노드/관계를 복구 적재합니다."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 SENT_TO 관계를 삭제하고 재적재합니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH,
        help=f"UNWIND 배치 크기 (기본값: {_DEFAULT_BATCH})",
    )
    args = parser.parse_args()
    repair(force=args.force, batch_size=args.batch_size)
