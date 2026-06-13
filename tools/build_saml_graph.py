"""SAML-D 기반 자금세탁 탐지 그래프 빌드 (P0 정합성 + P1 실네트워크 + P2 모티프).

PaySim은 자금세탁 네트워크가 없어(94.8% 단발거래) GNN이 무의미했다. SAML-D는
실제 세탁 유형(Structuring/Smurfing/Layered Fan-In·Out/Bipartite 등)을 담은
합성 AML 데이터셋(950만 거래, 계좌 90%가 3+회 거래, 경유 10.5%)이라 GNN에 적합하다.

서브그래프 설계 (AuraDB 한도 + 학습 속도 고려, 재현 가능 seed=42):
  - 세탁 거래 전체 보존 → 유형 구조(팬인/아웃·레이어링) 그대로 유지
  - 세탁 계좌의 정상 거래도 일부 포함(계좌당 cap) → '세탁 엣지=세탁계좌' 누수 방지
  - 정상 배경 거래 무작위 샘플 → 음성 클래스 구조

산출물 (모두 한 df_sampled에서 일관 생성 — node_idx↔계좌 정합성 보장, P0):
  model/saml_graph.pt          : x, edge_index, edge_attr, y, train/val/test_mask
  model/saml_node_mapping.csv  : node_idx, account_id, y  (loader가 그대로 사용)
  model/saml_feature_names.json: 노드 피처명 + 엣지 피처명
  model/saml_node_scaler.pkl   : 노드 피처 StandardScaler (train fit)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

SEED = 42
NORMAL_BG_SAMPLE = 100_000        # 정상 배경 거래 샘플 수
LAUNDER_ACCT_NORMAL_SAMPLE = 40_000  # 세탁 계좌 관련 정상 거래 샘플 (혼합 행태 부여)

NODE_FEATURES = [
    "out_count", "in_count",
    "out_amount_log", "in_amount_log",
    "out_mean_log", "in_mean_log",
    "unique_receivers", "unique_senders",
    "cross_border_ratio", "currency_mismatch_ratio",
    "passthrough_ratio",
]
EDGE_FEATURES = ["log_amount", "ptype_enc", "cross_border", "currency_mismatch", "time_norm"]


def _load() -> pd.DataFrame:
    print("[1/7] SAML-D 로드 중...")
    df = pd.read_csv(
        ROOT / "dataset" / "SAML-D.csv",
        usecols=["Time", "Date", "Sender_account", "Receiver_account", "Amount",
                 "Payment_currency", "Received_currency", "Sender_bank_location",
                 "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"],
        dtype={"Sender_account": str, "Receiver_account": str,
               "Payment_currency": "category", "Received_currency": "category",
               "Sender_bank_location": "category", "Receiver_bank_location": "category",
               "Payment_type": "category", "Laundering_type": "category"},
    )
    print(f"      전체 거래 {len(df):,}")
    return df


def _select_subgraph(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/7] 서브그래프 선택...")
    ltx = df[df.Is_laundering == 1]
    launder_accts = set(ltx.Sender_account) | set(ltx.Receiver_account)
    print(f"      세탁 거래 {len(ltx):,} / 세탁 연루 계좌 {len(launder_accts):,}")

    normal = df[df.Is_laundering == 0]

    # 세탁 계좌가 연루된 정상 거래 일부 (고정 샘플) — 현실적 혼합 행태 부여(누수 방지)
    la_normal = normal[normal.Sender_account.isin(launder_accts)
                       | normal.Receiver_account.isin(launder_accts)]
    la_normal = la_normal.sample(n=min(LAUNDER_ACCT_NORMAL_SAMPLE, len(la_normal)),
                                 random_state=SEED)
    print(f"      세탁계좌 정상거래(샘플) {len(la_normal):,}")

    # 정상 배경 샘플
    bg = normal.sample(n=NORMAL_BG_SAMPLE, random_state=SEED)

    sub = pd.concat([ltx, la_normal, bg]).drop_duplicates().reset_index(drop=True)
    print(f"      서브그래프 거래 {len(sub):,}")
    return sub


def _build(sub: pd.DataFrame):
    print("[3/7] 계좌 인덱싱(node_idx↔계좌)...")
    accounts = pd.Index(pd.concat([sub.Sender_account, sub.Receiver_account]).unique())
    acc2idx = {a: i for i, a in enumerate(accounts)}
    num_nodes = len(accounts)
    sub["src"] = sub.Sender_account.map(acc2idx).astype(np.int64)
    sub["dst"] = sub.Receiver_account.map(acc2idx).astype(np.int64)
    print(f"      노드 {num_nodes:,} / 엣지 {len(sub):,}")

    print("[4/7] 파생 컬럼...")
    sub["log_amount"] = np.log1p(sub.Amount.astype(float))
    sub["cross_border"] = (sub.Sender_bank_location.astype(str)
                           != sub.Receiver_bank_location.astype(str)).astype(int)
    sub["currency_mismatch"] = (sub.Payment_currency.astype(str)
                                != sub.Received_currency.astype(str)).astype(int)
    sub["ptype_enc"] = sub.Payment_type.astype("category").cat.codes
    # 시간 정규화 (HH:MM:SS -> 0~1)
    t = pd.to_timedelta(sub.Time.astype(str), errors="coerce").dt.total_seconds().fillna(0)
    sub["time_norm"] = (t / 86400.0).clip(0, 1)

    print("[5/7] 노드 피처(네트워크/모티프)...")
    g_out = sub.groupby("src", observed=True)
    g_in = sub.groupby("dst", observed=True)
    feat = pd.DataFrame(index=range(num_nodes))
    feat["out_count"] = g_out.size().reindex(feat.index, fill_value=0)
    feat["in_count"] = g_in.size().reindex(feat.index, fill_value=0)
    feat["out_amount_log"] = np.log1p(g_out.Amount.sum().reindex(feat.index, fill_value=0))
    feat["in_amount_log"] = np.log1p(g_in.Amount.sum().reindex(feat.index, fill_value=0))
    feat["out_mean_log"] = np.log1p(g_out.Amount.mean().reindex(feat.index, fill_value=0))
    feat["in_mean_log"] = np.log1p(g_in.Amount.mean().reindex(feat.index, fill_value=0))
    feat["unique_receivers"] = g_out.dst.nunique().reindex(feat.index, fill_value=0)
    feat["unique_senders"] = g_in.src.nunique().reindex(feat.index, fill_value=0)
    feat["cross_border_ratio"] = g_out.cross_border.mean().reindex(feat.index).fillna(0)
    feat["currency_mismatch_ratio"] = g_out.currency_mismatch.mean().reindex(feat.index).fillna(0)
    out_amt = g_out.Amount.sum().reindex(feat.index, fill_value=0.0)
    in_amt = g_in.Amount.sum().reindex(feat.index, fill_value=0.0)
    feat["passthrough_ratio"] = (np.minimum(out_amt, in_amt) / (np.maximum(out_amt, in_amt) + 1.0))
    x_raw = feat[NODE_FEATURES].values.astype(np.float32)

    print("[6/7] 라벨 + 분할 + 스케일...")
    launder_accts = set(sub[sub.Is_laundering == 1].Sender_account) \
        | set(sub[sub.Is_laundering == 1].Receiver_account)
    y = np.array([1 if a in launder_accts else 0 for a in accounts], dtype=np.int64)
    print(f"      세탁 노드 {y.sum():,} ({y.mean()*100:.2f}%)")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(num_nodes)
    n_tr, n_va = int(num_nodes * 0.7), int(num_nodes * 0.1)
    train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    masks = {}
    for nm, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        m = np.zeros(num_nodes, dtype=bool); m[idx] = True; masks[nm] = m

    scaler = StandardScaler().fit(x_raw[train_idx])
    x = scaler.transform(x_raw).astype(np.float32)

    # 사람이 읽을 수 있는 원본(raw) 노드 피처 — Neo4j 속성/GraphRAG 증거용
    raw_df = pd.DataFrame(x_raw, columns=NODE_FEATURES)
    raw_df.insert(0, "node_idx", range(num_nodes))
    raw_df.to_csv(ROOT / "model" / "saml_node_features_raw.csv", index=False, encoding="utf-8")

    edge_index = torch.tensor([sub.src.values, sub.dst.values], dtype=torch.long)
    edge_attr = torch.tensor(sub[EDGE_FEATURES].values.astype(np.float32))
    edge_is_launder = torch.tensor(sub.Is_laundering.values.astype(np.int64))

    graph = {
        "x": torch.tensor(x), "edge_index": edge_index, "edge_attr": edge_attr,
        "edge_is_laundering": edge_is_launder,
        "y": torch.tensor(y),
        "train_mask": torch.tensor(masks["train"]),
        "val_mask": torch.tensor(masks["val"]),
        "test_mask": torch.tensor(masks["test"]),
    }
    mapping = pd.DataFrame({"node_idx": range(num_nodes),
                            "account_id": accounts.astype(str), "y": y})
    return graph, mapping, scaler, sub


def main():
    df = _load()
    sub = _select_subgraph(df)
    del df
    graph, mapping, scaler, sub = _build(sub)

    print("[7/7] 저장...")
    mdir = ROOT / "model"
    torch.save(graph, mdir / "saml_graph.pt")
    mapping.to_csv(mdir / "saml_node_mapping.csv", index=False, encoding="utf-8")
    joblib.dump(scaler, mdir / "saml_node_scaler.pkl")
    json.dump({"node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES},
              open(mdir / "saml_feature_names.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # 세탁 거래 서브셋(엣지 속성+유형)도 저장 — DB 적재/증거용
    sub_cols = ["Sender_account", "Receiver_account", "Amount", "Payment_type",
                "Sender_bank_location", "Receiver_bank_location",
                "Payment_currency", "Received_currency", "Date", "Time",
                "Is_laundering", "Laundering_type", "src", "dst"]
    sub[sub_cols].to_parquet(mdir / "saml_edges.parquet", index=False) if _has_parquet() \
        else sub[sub_cols].to_csv(mdir / "saml_edges.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  완료: 노드 {graph['x'].shape[0]:,} / 엣지 {graph['edge_index'].shape[1]:,}")
    print(f"  세탁 노드 {int(graph['y'].sum()):,} ({graph['y'].float().mean()*100:.2f}%)")
    print(f"  노드 피처 {graph['x'].shape[1]}개 / 엣지 피처 {graph['edge_attr'].shape[1]}개")
    print("=" * 60)


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
