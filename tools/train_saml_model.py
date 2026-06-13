"""SAML-D 그래프로 GraphSAGE(임베딩) + XGBoost 하이브리드 모델 재학습.

build_saml_graph.py 산출물(model/saml_graph.pt)을 입력으로:
  1. EmbeddingExtractor(GraphSAGE 3층) 분류 학습 → 64차원 노드 임베딩
  2. 하이브리드 피처 [노드피처 11 + 임베딩 64] (순서: [orig, emb] — 추론 헬퍼와 동일)
  3. 74-dim... 75-dim StandardScaler(train fit) + XGBoost 학습
  4. 저장: model/saml_gnn.pth, model/saml_fraud_model.pkl (xgb_model, scaler, all_feature_names)
검증: test AUC/AP + 고확신 정밀도(보정).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from models.embedding_extractor import EmbeddingExtractor

SEED = 42
EPOCHS = 120
HIDDEN = 128
EMBED = 64


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    g = torch.load(ROOT / "model" / "saml_graph.pt", map_location="cpu")
    x, ei, y = g["x"], g["edge_index"], g["y"]
    trm, vm, tem = g["train_mask"], g["val_mask"], g["test_mask"]
    in_dim = x.shape[1]
    node_feats = json.load(open(ROOT / "model" / "saml_feature_names.json", encoding="utf-8"))["node_features"]
    print(f"[학습] 노드 {x.shape[0]:,} / 엣지 {ei.shape[1]:,} / 피처 {in_dim} / 세탁 {int(y.sum()):,}")

    # ── GNN 학습 (분류 헤드) ─────────────────────────────────────────
    model = EmbeddingExtractor(in_channels=in_dim, hidden_channels=HIDDEN, embed_dim=EMBED)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    pos_w = float((y[trm] == 0).sum()) / float(max((y[trm] == 1).sum(), 1))
    weight = torch.tensor([1.0, pos_w])
    crit = torch.nn.CrossEntropyLoss(weight=weight)

    print(f"[GNN] 학습 시작 (epochs={EPOCHS}, pos_weight={pos_w:.1f})")
    for ep in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        out = model.classify(x, ei)
        loss = crit(out[trm], y[trm])
        loss.backward(); opt.step()
        if ep % 20 == 0:
            model.eval()
            with torch.no_grad():
                pv = F.softmax(model.classify(x, ei), dim=1)[:, 1].numpy()
            auc = roc_auc_score(y[vm].numpy(), pv[vm.numpy()])
            print(f"  epoch {ep:3d} | loss {loss.item():.4f} | val AUC {auc:.3f}")

    # ── 임베딩 추출 ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        emb = model(x, ei).numpy().astype(np.float32)
    print(f"[GNN] 임베딩 {emb.shape}")

    # ── 하이브리드 [노드피처, 임베딩] + 스케일 + XGBoost ────────────
    x_np = x.numpy().astype(np.float32)
    hybrid = np.hstack([x_np, emb]).astype(np.float32)
    feat_names = list(node_feats) + [f"GNN_Emb_{i}" for i in range(EMBED)]

    tr = trm.numpy(); va = vm.numpy(); te = tem.numpy(); y_np = y.numpy()
    scaler = StandardScaler().fit(hybrid[tr])
    Xs = scaler.transform(hybrid).astype(np.float32)

    pos_w_xgb = (y_np[tr] == 0).sum() / max((y_np[tr] == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=pos_w_xgb, random_state=SEED, n_jobs=-1, eval_metric="aucpr",
        early_stopping_rounds=40,
    )
    clf.fit(Xs[tr], y_np[tr], eval_set=[(Xs[va], y_np[va])], verbose=False)

    p = clf.predict_proba(Xs)[:, 1]
    print("\n=== 하이브리드(GNN+XGBoost) 성능 ===")
    for nm, m in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm}: AUC={roc_auc_score(y_np[m], p[m]):.3f} "
              f"AP={average_precision_score(y_np[m], p[m]):.3f}")
    print("  고확신 정밀도(test):")
    for th in [0.7, 0.9, 0.99]:
        sel = te & (p >= th)
        n = sel.sum()
        print(f"    prob>={th}: {n:,}개 정밀도 {y_np[sel].mean()*100:.1f}%" if n else f"    prob>={th}: 0개")

    # ── 모티프 변별력 (XGBoost 피처 중요도 상위) ──────────────────────
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:8]
    print("  상위 기여 피처:", [(feat_names[i], round(float(imp[i]), 3)) for i in order])

    # ── 저장 ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), ROOT / "model" / "saml_gnn.pth")
    joblib.dump({"xgb_model": clf, "scaler": scaler, "all_feature_names": feat_names},
                ROOT / "model" / "saml_fraud_model.pkl")
    print("\n[저장] model/saml_gnn.pth, model/saml_fraud_model.pkl")


if __name__ == "__main__":
    main()
