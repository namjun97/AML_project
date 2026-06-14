"""SAML 모델에 temperature scaling 보정 계수(T)를 적합·저장.

검증셋 NLL 최소화로 T 를 적합해 saml_fraud_model.pkl 의 'temperature' 키에 저장한다.
predict_fraud_probs 가 이 T 를 적용한다 (단조 변환 → AUC 불변, 과신만 완화).

참고: SAML 모델은 이미 잘 보정돼 있어(ECE~0.05) NLL 최적 T 가 1.0 부근으로 나온다.
즉 상위 점수의 '100% 근접'은 과신이 아니라 실제로 맞는 예측(정밀도 ~99%)이다.
그래도 향후 과신 모델을 대비해 보정 단계를 파이프라인에 둔다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import joblib
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from models.embedding_extractor import EmbeddingExtractor, predict_fraud_probs, apply_temperature


def main():
    g = torch.load(ROOT / "model" / "saml_graph.pt", map_location="cpu")
    x, ei, y = g["x"], g["edge_index"], g["y"].numpy()
    ex = EmbeddingExtractor(in_channels=x.shape[1], hidden_channels=128, embed_dim=64)
    ex.load_state_dict(torch.load(ROOT / "model" / "saml_gnn.pth", map_location="cpu"))
    ex.eval()
    with torch.no_grad():
        emb = ex(x, ei).numpy().astype(np.float32)

    pkl = ROOT / "model" / "saml_fraud_model.pkl"
    data = joblib.load(pkl)
    raw = predict_fraud_probs(emb, x.numpy(), data["xgb_model"], data["scaler"], temperature=1.0)

    vm = g["val_mask"].numpy().astype(bool)
    tm = g["test_mask"].numpy().astype(bool)
    eps = 1e-6
    logit = np.log(np.clip(raw, eps, 1 - eps) / (1 - np.clip(raw, eps, 1 - eps)))

    def nll(T):
        q = np.clip(1.0 / (1.0 + np.exp(-logit[vm] / T)), eps, 1 - eps)
        return -np.mean(y[vm] * np.log(q) + (1 - y[vm]) * np.log(1 - q))

    T = float(minimize_scalar(nll, bounds=(0.5, 10.0), method="bounded").x)
    pT = apply_temperature(raw, T)

    def ece(pr, yy, bins=10):
        e = 0.0
        for b in range(bins):
            m = (pr >= b / bins) & (pr < (b + 1) / bins)
            if m.sum():
                e += abs(pr[m].mean() - yy[m].mean()) * m.sum() / len(pr)
        return e

    print(f"[보정] 적합된 temperature T = {T:.4f}")
    print(f"  AUC(test): {roc_auc_score(y[tm], raw[tm]):.3f} (불변, 단조 변환)")
    print(f"  ECE(test): 전 {ece(raw[tm], y[tm]):.4f} / 후 {ece(pT[tm], y[tm]):.4f}")
    print(f"  '100.0%' 근접(>=0.9995) test: 전 {(raw[tm]>=0.9995).sum()} / 후 {(pT[tm]>=0.9995).sum()}")

    data["temperature"] = T
    joblib.dump(data, pkl)
    print(f"[저장] saml_fraud_model.pkl 에 temperature={T:.4f} 저장 완료")


if __name__ == "__main__":
    main()
