"""SAML-D 그래프를 Neo4j(AuraDB)에 적재 — 정합성(P0) 보장 + 네트워크 증거.

build_saml_graph.py / train_saml_model.py 산출물만 사용하므로, DB의 node_idx·계좌·
라벨·모티프 속성·임베딩 기반 fraud_prob 가 모두 '하나의 df_sampled'에서 일관 생성된다
(node_idx↔계좌 어긋남 원천 차단 — y == is_laundering 100%).

노드 속성: account_id, node_idx, is_laundering, fraud_prob + 모티프(out/in_count,
  unique_receivers/senders, out/in_amount, cross_border_ratio, currency_mismatch_ratio,
  passthrough_ratio)
엣지 속성: amount, payment_type, sender/receiver_location, cross_border, currency_mismatch,
  is_laundering_tx, laundering_type, date

사용법: python tools/neo4j_load_saml.py --force
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from neo4j_config import get_driver
from models.embedding_extractor import EmbeddingExtractor, predict_fraud_probs

_BATCH = 2000


def _compute_probs():
    g = torch.load(ROOT / "model" / "saml_graph.pt", map_location="cpu")
    x, ei = g["x"], g["edge_index"]
    ex = EmbeddingExtractor(in_channels=x.shape[1], hidden_channels=128, embed_dim=64)
    ex.load_state_dict(torch.load(ROOT / "model" / "saml_gnn.pth", map_location="cpu"))
    ex.eval()
    with torch.no_grad():
        emb = ex(x, ei).numpy().astype(np.float32)
    data = joblib.load(ROOT / "model" / "saml_fraud_model.pkl")
    probs = predict_fraud_probs(emb, x.numpy(), data["xgb_model"], data["scaler"])
    return probs


def load(force: bool = False, batch: int = _BATCH):
    t0 = time.time()
    mapping = pd.read_csv(ROOT / "model" / "saml_node_mapping.csv",
                          dtype={"account_id": str})
    raw = pd.read_csv(ROOT / "model" / "saml_node_features_raw.csv")
    edges = pd.read_parquet(ROOT / "model" / "saml_edges.parquet")
    probs = _compute_probs()
    print(f"[SAML] 노드 {len(mapping):,} / 엣지 {len(edges):,} / fraud_prob 평균 {probs.mean():.4f}")

    m = mapping.merge(raw, on="node_idx")
    m["fraud_prob"] = probs[m.node_idx.values].round(6)

    node_rows = [{
        "account_id": str(r.account_id), "node_idx": int(r.node_idx),
        "is_laundering": bool(r.y), "fraud_prob": float(r.fraud_prob),
        "out_count": int(r.out_count), "in_count": int(r.in_count),
        "unique_receivers": int(r.unique_receivers), "unique_senders": int(r.unique_senders),
        "out_amount": round(float(np.expm1(r.out_amount_log)), 2),
        "in_amount": round(float(np.expm1(r.in_amount_log)), 2),
        "cross_border_ratio": round(float(r.cross_border_ratio), 4),
        "currency_mismatch_ratio": round(float(r.currency_mismatch_ratio), 4),
        "passthrough_ratio": round(float(r.passthrough_ratio), 4),
    } for r in m.itertuples(index=False)]

    edge_rows = [{
        "src": str(e.Sender_account), "dst": str(e.Receiver_account),
        "amount": round(float(e.Amount), 2), "payment_type": str(e.Payment_type),
        "sender_location": str(e.Sender_bank_location),
        "receiver_location": str(e.Receiver_bank_location),
        "cross_border": str(e.Sender_bank_location) != str(e.Receiver_bank_location),
        "currency_mismatch": str(e.Payment_currency) != str(e.Received_currency),
        "is_laundering_tx": bool(e.Is_laundering),
        "laundering_type": str(e.Laundering_type),
        "date": str(e.Date),
    } for e in edges.itertuples(index=False)]

    d = get_driver()
    print("[SAML] Neo4j 연결 성공")
    with d.session() as s:
        cnt = s.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        if cnt > 0 and not force:
            print(f"[SAML] 기존 {cnt:,} 노드 존재 — --force 로 재적재")
            d.close(); return
        if force and cnt > 0:
            print(f"[SAML] 기존 데이터 삭제 중...")
            while s.run("MATCH (n) WITH n LIMIT 20000 DETACH DELETE n RETURN count(*) AS c").single()["c"]:
                pass
        s.run("CREATE INDEX account_id_idx IF NOT EXISTS FOR (a:Account) ON (a.account_id)")
        s.run("CREATE INDEX account_node_idx2 IF NOT EXISTS FOR (a:Account) ON (a.node_idx)")

        for k in range(0, len(node_rows), batch):
            s.run("""
                UNWIND $rows AS r
                MERGE (a:Account {account_id: r.account_id})
                SET a += r
            """, rows=node_rows[k:k + batch])
            print(f"\r  노드 {min(k+batch,len(node_rows)):,}/{len(node_rows):,}", end="", flush=True)
        print()
        for k in range(0, len(edge_rows), batch):
            s.run("""
                UNWIND $rows AS r
                MATCH (a:Account {account_id: r.src})
                MATCH (b:Account {account_id: r.dst})
                CREATE (a)-[:SENT_TO {
                    amount: r.amount, payment_type: r.payment_type,
                    sender_location: r.sender_location, receiver_location: r.receiver_location,
                    cross_border: r.cross_border, currency_mismatch: r.currency_mismatch,
                    is_laundering_tx: r.is_laundering_tx, laundering_type: r.laundering_type,
                    date: r.date
                }]->(b)
            """, rows=edge_rows[k:k + batch])
            print(f"\r  엣지 {min(k+batch,len(edge_rows)):,}/{len(edge_rows):,}", end="", flush=True)
        print()

    with d.session() as s:
        n = s.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        r = s.run("MATCH ()-[x:SENT_TO]->() RETURN count(x) AS c").single()["c"]
        ln = s.run("MATCH (a:Account {is_laundering:true}) RETURN count(a) AS c").single()["c"]
        lt = s.run("MATCH ()-[x:SENT_TO]->() WHERE x.is_laundering_tx RETURN count(x) AS c").single()["c"]
    d.close()
    print("\n" + "=" * 60)
    print(f"  SAML-D 적재 완료 ({time.time()-t0:.1f}s)")
    print(f"  [OK] 계좌 {n:,} / 세탁 계좌 {ln:,}")
    print(f"  [OK] 거래 {r:,} / 세탁 거래 {lt:,}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=_BATCH)
    a = ap.parse_args()
    load(force=a.force, batch=a.batch_size)
