from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 프로젝트 루트 경로 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.embedding_extractor import EmbeddingExtractor
from neo4j_config import get_driver


# 상수 — aml_project.ipynb 동일한 값 유지 (재현성)

_TRANSACTION_TYPES  = ["TRANSFER", "CASH_OUT"]
_NORMAL_SAMPLE_SIZE = 50_000
_RANDOM_STATE       = 42

_FEATURE_COLS = [
    "send_count", "send_total", "send_mean", "send_max",
    "zero_balance_cnt", "mismatch_sum",
    "recv_count", "recv_total", "recv_mean", "empty_acct_recv",
]

_EDGE_FEATURE_COLS = [
    "log_amount", "type_encoded", "balance_diff_orig",
    "balance_diff_dest", "balance_mismatch", "hour_of_day_norm",
]

# edge_attr 인덱스
_IDX_LOG_AMOUNT       = 0
_IDX_TYPE_ENCODED     = 1
_IDX_BALANCE_MISMATCH = 4
_IDX_HOUR_NORM        = 5

_TYPE_MAP = {0: "TRANSFER", 1: "CASH_OUT"}

_DEFAULT_BATCH = 1_000

# 1. df_sampled 재구성 + account 인덱스 매핑 복원

def _rebuild_df_sampled(csv_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    aml_project.ipynb와 동일한 전처리로 df_sampled를 재구성하고
    LabelEncoder(계좌명 ↔ 노드 인덱스 매핑)를 반환합니다.

    Returns:
        (df_sampled, le)
        df_sampled: 전처리된 거래 DataFrame
        le        : LabelEncoder (le.classes_[i] = i번 노드의 계좌명)
    """
    print("[Loader] CSV 로드 중...")
    df = pd.read_csv(csv_path)

    # 1) TRANSFER / CASH_OUT 필터
    df = df[df["type"].isin(_TRANSACTION_TYPES)].copy()

    # 2) 파생 컬럼
    df["log_amount"]       = np.log1p(df["amount"])
    df["type_encoded"]     = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1})
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["balance_mismatch"]  = (df["newbalanceOrig"] != df["oldbalanceOrg"] - df["amount"]).astype(int)
    df["orig_zero_balance"] = (df["newbalanceOrig"] == 0).astype(int)
    df["hour_of_day_norm"]  = (df["step"] % 24) / 23.0

    # 3) 사기 전체 + 정상 50000 샘플
    fraud_df  = df[df["isFraud"] == 1]
    normal_df = df[df["isFraud"] == 0].sample(
        n=min(_NORMAL_SAMPLE_SIZE, len(df[df["isFraud"] == 0])),
        random_state=_RANDOM_STATE,
    )
    df_sampled = pd.concat([fraud_df, normal_df]).reset_index(drop=True)

    # 4) LabelEncoder (node_idx ↔ account_name)
    all_accounts = pd.concat(
        [df_sampled["nameOrig"], df_sampled["nameDest"]]
    ).unique()
    le = LabelEncoder()
    le.fit(all_accounts)

    df_sampled["src_idx"] = le.transform(df_sampled["nameOrig"])
    df_sampled["dst_idx"] = le.transform(df_sampled["nameDest"])

    print(
        f"[Loader] df_sampled: {len(df_sampled):,}건 / "
        f"계좌 수: {len(le.classes_):,}개"
    )
    return df_sampled, le

# 2. 노드 피처 DataFrame 재구성 (원본 비스케일)

def _build_account_df(df_sampled: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame: # 계좌별 집계 피처

    # 송금자 관점
    orig_feat = df_sampled.groupby("nameOrig").agg(
        send_count       =("log_amount", "count"),
        send_total       =("log_amount", "sum"),
        send_mean        =("log_amount", "mean"),
        send_max         =("log_amount", "max"),
        zero_balance_cnt =("orig_zero_balance", "sum"),
        mismatch_sum     =("balance_mismatch", "sum"),
    ).reset_index().rename(columns={"nameOrig": "account"})

    # 수취자 관점
    dest_feat = df_sampled.groupby("nameDest").agg(
        recv_count     =("log_amount", "count"),
        recv_total     =("log_amount", "sum"),
        recv_mean      =("log_amount", "mean"),
        empty_acct_recv=("balance_mismatch", "sum"),
    ).reset_index().rename(columns={"nameDest": "account"})

    account_df = pd.DataFrame({"account": le.classes_})
    account_df = account_df.merge(orig_feat, on="account", how="left")
    account_df = account_df.merge(dest_feat, on="account", how="left")
    account_df = account_df.fillna(0)

    # 노드 인덱스 (le.classes_ 순서 = 노드 인덱스)
    account_df["node_idx"] = np.arange(len(account_df))

    # 사기 라벨
    fraud_senders   = set(df_sampled[df_sampled["isFraud"] == 1]["nameOrig"])
    fraud_receivers = set(df_sampled[df_sampled["isFraud"] == 1]["nameDest"])
    account_df["is_fraud"] = account_df["account"].isin(
        fraud_senders | fraud_receivers
    ).astype(int)

    return account_df # 스케일링 전 원본 데이터 Neo4j에 저장

# 3. GNN 임베딩 계산

def _compute_embeddings(
    graph_dict: dict,
    gnn_path: Path,
) -> np.ndarray:

    print("[Loader] GNN 임베딩 계산 중...")
    in_dim = graph_dict["x"].shape[1]
    extractor = EmbeddingExtractor(
        in_channels=in_dim,
        hidden_channels=128,
        embed_dim=64,
    )
    extractor.load_state_dict(torch.load(str(gnn_path), map_location="cpu")) # 64차원 계산
    extractor.eval()

    with torch.no_grad():
        embs = extractor(graph_dict["x"], graph_dict["edge_index"])

    print(f"[Loader] 임베딩 shape: {embs.shape}")
    return embs.numpy()


# 4. XGBoost 사기 확률 계산

def _compute_fraud_probs(  # 전체 노드 사기 확률 계산
    embs: np.ndarray,
    graph_dict: dict,
    xgb_path: Path,
) -> np.ndarray:
    print("[Loader] XGBoost 사기 확률 계산 중...")
    xgb_data  = joblib.load(str(xgb_path))
    xgb_model = xgb_data["xgb_model"]

    orig_features = graph_dict["x"].numpy()   # 스케일된 값 사용 (학습 시 동일)
    X_all = np.hstack([embs, orig_features])

    probs = xgb_model.predict_proba(X_all)[:, 1]
    print(f"[Loader] 사기 확률 계산 완료  (평균: {probs.mean():.4f})")
    return probs


# 5. Neo4j 배치 적재

def _load_nodes(
    session,
    account_df: pd.DataFrame,
    probs: np.ndarray,
    batch_size: int = _DEFAULT_BATCH,
) -> None:
    """
    (:Account) 노드를 UNWIND 배치로 적재합니다.
    """
    print("[Loader] 노드 적재 시작...")
    rows = []
    for i, row in account_df.iterrows():
        rows.append({
            "account_id"     : str(row["account"]),
            "node_idx"       : int(row["node_idx"]),
            "is_fraud"       : bool(row["is_fraud"]),
            "fraud_prob"     : round(float(probs[int(row["node_idx"])]), 6),
            "send_count"     : float(row.get("send_count", 0)),
            "send_total"     : float(row.get("send_total", 0)),
            "send_mean"      : float(row.get("send_mean", 0)),
            "send_max"       : float(row.get("send_max", 0)),
            "recv_count"     : float(row.get("recv_count", 0)),
            "recv_total"     : float(row.get("recv_total", 0)),
            "recv_mean"      : float(row.get("recv_mean", 0)),
            "zero_balance_cnt": float(row.get("zero_balance_cnt", 0)),
            "empty_acct_recv" : float(row.get("empty_acct_recv", 0)),
            "mismatch_sum"   : float(row.get("mismatch_sum", 0)),
        })

    total  = len(rows)
    loaded = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        session.run(
            """
            UNWIND $rows AS r
            MERGE (a:Account {account_id: r.account_id})
            SET   a.node_idx        = r.node_idx,
                  a.is_fraud        = r.is_fraud,
                  a.fraud_prob      = r.fraud_prob,
                  a.send_count      = r.send_count,
                  a.send_total      = r.send_total,
                  a.send_mean       = r.send_mean,
                  a.send_max        = r.send_max,
                  a.recv_count      = r.recv_count,
                  a.recv_total      = r.recv_total,
                  a.recv_mean       = r.recv_mean,
                  a.zero_balance_cnt = r.zero_balance_cnt,
                  a.empty_acct_recv  = r.empty_acct_recv,
                  a.mismatch_sum    = r.mismatch_sum
            """,
            rows=batch,
        )
        loaded += len(batch)
        print(f"\r  노드 적재: {loaded:,}/{total:,}", end="", flush=True)

    print(f"\r  노드 적재 완료: {loaded:,}개          ")


def _load_edges(  # 데이터베이스 세션, 샘플링된 데이터프레임, 라벨 인코더, batch 사이즈를 인자로 받는 함수
    session,
    df_sampled: pd.DataFrame,
    le: LabelEncoder,
    batch_size: int = _DEFAULT_BATCH,
) -> None:
    print("[Loader] 관계(엣지) 적재 시작...")

    rows: list[dict[str, Any]] = []
    for _, tx in df_sampled.iterrows():
        amount_orig = float(np.expm1(tx["log_amount"])) # 역로그 변환(로그 변환된 금액을 실제 금액 단위로 되돌림)
        rows.append({
            "src_id"           : str(tx["nameOrig"]),
            "dst_id"           : str(tx["nameDest"]),
            "amount"           : round(amount_orig, 2),
            "log_amount"       : round(float(tx["log_amount"]), 6),
            "tx_type"          : str(tx["type"]),
            "step"             : int(tx["step"]),
            "balance_mismatch" : int(tx["balance_mismatch"]),
            "hour_of_day_norm" : round(float(tx["hour_of_day_norm"]), 4),
            "is_fraud_tx"      : bool(tx["isFraud"]),
        })

    total  = len(rows)
    loaded = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        session.run(
            """
            UNWIND $rows AS r
            MATCH (src:Account {account_id: r.src_id})
            MATCH (dst:Account {account_id: r.dst_id})
            CREATE (src)-[:SENT_TO {
                amount          : r.amount,
                log_amount      : r.log_amount,
                tx_type         : r.tx_type,
                step            : r.step,
                balance_mismatch: r.balance_mismatch,
                hour_of_day_norm: r.hour_of_day_norm,
                is_fraud_tx     : r.is_fraud_tx
            }]->(dst)
            """,
            rows=batch,
        )
        loaded += len(batch)
        print(f"\r  관계 적재: {loaded:,}/{total:,}", end="", flush=True)

    print(f"\r  관계 적재 완료: {loaded:,}개          ")


# ======================================================================
# 6. 메인 진입점
# ======================================================================

def load_graph_to_neo4j(
    csv_path: str  = "dataset/paysim1.csv",           # paysim1.csv 경로
    graph_path: str = "model/processed_graph_data.pt",# processed_graph_data.pt 경로
    gnn_path: str  = "gnn_model.pth",                 # gnn_model.pth 경로
    xgb_path: str  = "fraud_model.pkl",               # fraud_model.pkl 경로
    force: bool    = False,                           # True면 기존 데이터 삭제 후 재적재
    batch_size: int = _DEFAULT_BATCH,                 # Unwind 배치 크기(default 1000)
) -> None:
    t0 = time.time()

    root = ROOT

    # ── 경로 절대화 ─────────────────────────────────────────────────
    csv_path_abs   = root / csv_path
    graph_path_abs = root / graph_path
    gnn_path_abs   = root / gnn_path
    xgb_path_abs   = root / xgb_path

    for p in [csv_path_abs, graph_path_abs, gnn_path_abs, xgb_path_abs]:
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {p}")

    # ── Neo4j 연결 ───────────────────────────────────────────────────
    driver = get_driver()
    print("[Loader] Neo4j 연결 성공")

    with driver.session() as session:
        node_count = session.run(
            "MATCH (a:Account) RETURN count(a) AS cnt"
        ).single()["cnt"]

        if node_count > 0 and not force:
            print(
                f"[Loader] 이미 {node_count:,}개 노드가 존재합니다. "
                "--force 옵션으로 재적재할 수 있습니다."
            )
            driver.close()
            return

        if force and node_count > 0:
            print(f"[Loader] 기존 {node_count:,}개 노드 삭제 중...")
            session.run("MATCH (n) DETACH DELETE n")
            print("[Loader] 삭제 완료.")

    # ── 데이터 준비 ──────────────────────────────────────────────────
    df_sampled, le       = _rebuild_df_sampled(csv_path_abs)
    account_df           = _build_account_df(df_sampled, le)

    graph_dict = torch.load(str(graph_path_abs), map_location="cpu")
    embs       = _compute_embeddings(graph_dict, gnn_path_abs)
    probs      = _compute_fraud_probs(embs, graph_dict, xgb_path_abs)

    # ── Neo4j 적재 ───────────────────────────────────────────────────
    with driver.session() as session:
        _load_nodes(session, account_df, probs, batch_size)
        _load_edges(session, df_sampled, le, batch_size)

    # ── 결과 검증 ────────────────────────────────────────────────────
    with driver.session() as session:
        n = session.run("MATCH (a:Account) RETURN count(a) AS cnt").single()["cnt"]
        r = session.run("MATCH ()-[r:SENT_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        fraud_n = session.run(
            "MATCH (a:Account {is_fraud: true}) RETURN count(a) AS cnt"
        ).single()["cnt"]
        high_risk = session.run(
            "MATCH (a:Account) WHERE a.fraud_prob > 0.7 RETURN count(a) AS cnt"
        ).single()["cnt"]

    driver.close()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print(f"  Neo4j 적재 완료 ({elapsed:.1f}초)")
    print(f"  [OK] 계좌 노드   : {n:,}개")
    print(f"  [OK] 거래 관계   : {r:,}개")
    print(f"  [!]  사기 계좌   : {fraud_n:,}개")
    print(f"  [!]  고위험(>70%): {high_risk:,}개")
    print("=" * 60)


# ======================================================================
# CLI 진입점
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PaySim AML 그래프를 Neo4j에 적재합니다."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 데이터를 삭제하고 재적재합니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH,
        help=f"UNWIND 배치 크기 (기본값: {_DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--csv",
        default="dataset/paysim1.csv",
        help="paysim1.csv 경로",
    )
    args = parser.parse_args()

    load_graph_to_neo4j(
        csv_path=args.csv,
        force=args.force,
        batch_size=args.batch_size,
    )
