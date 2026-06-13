"""XGBoost 하이브리드 추론 파이프라인 회귀 테스트.

과거 버그: 서빙 3경로(app/resource_loader/neo4j_loader)가 모두
  - 피처 순서를 [임베딩64, 원본10] 으로 뒤집고
  - 학습 시 적용한 74-dim StandardScaler 를 생략
하여 XGBoost 입력이 학습(aml_model_comparison.ipynb: [원본10, 임베딩64] + scaler)과
어긋났고, fraud_prob 의 test AUC 가 0.34(무작위 이하)로 무의미했다.

이 테스트는 (1) 헬퍼의 순서/스케일러 적용을 단위 검증하고,
(2) 실제 아티팩트가 있으면 test AUC > 0.9 를 보장해 동일 회귀를 차단한다.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from models.embedding_extractor import (
    build_hybrid_features,
    hybrid_feature_names,
    predict_fraud_probs,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. 순수 단위 테스트 (아티팩트 불필요)
# ---------------------------------------------------------------------------

class TestBuildHybridFeatures:
    def test_orig_features_come_first(self):
        """학습과 동일하게 [원본, 임베딩] 순서로 결합한다."""
        orig = np.array([[1.0, 2.0]])         # (1, 2)
        emb  = np.array([[10.0, 20.0, 30.0]])  # (1, 3)
        out = build_hybrid_features(orig, emb)
        assert out.shape == (1, 5)
        assert list(out[0]) == [1.0, 2.0, 10.0, 20.0, 30.0]

    def test_output_is_float32(self):
        out = build_hybrid_features(np.array([[1]]), np.array([[2]]))
        assert out.dtype == np.float32


class TestHybridFeatureNames:
    def test_reorders_orig_before_embeddings(self):
        """저장된 [GNN_Emb*, 원본*] 를 [원본*, GNN_Emb*] 로 되돌린다."""
        stored = ["GNN_Emb_0", "GNN_Emb_1", "send_count", "mismatch_sum"]
        out = hybrid_feature_names(stored)
        assert out == ["send_count", "mismatch_sum", "GNN_Emb_0", "GNN_Emb_1"]

    def test_preserves_relative_order_within_groups(self):
        stored = ["GNN_Emb_5", "GNN_Emb_2", "b_feat", "a_feat"]
        out = hybrid_feature_names(stored)
        # 그룹 내 상대 순서 유지
        assert out == ["b_feat", "a_feat", "GNN_Emb_5", "GNN_Emb_2"]


class TestPredictFraudProbs:
    def test_applies_scaler_and_order(self):
        """[원본,임베딩] 결합 후 scaler.transform 을 거쳐 predict_proba 에 전달."""
        orig = np.array([[1.0, 2.0]])
        emb  = np.array([[3.0, 4.0]])

        scaler = MagicMock()
        scaler.transform.side_effect = lambda X: X * 10.0  # 적용 여부 추적

        xgb = MagicMock()
        xgb.predict_proba.return_value = np.array([[0.2, 0.8]])

        out = predict_fraud_probs(emb, orig, xgb, scaler)

        # scaler 가 [원본, 임베딩] 순서 행렬을 받았는지
        scaler_input = scaler.transform.call_args[0][0]
        assert list(scaler_input[0]) == [1.0, 2.0, 3.0, 4.0]
        # predict_proba 가 스케일된 값을 받았는지
        proba_input = xgb.predict_proba.call_args[0][0]
        assert list(proba_input[0]) == [10.0, 20.0, 30.0, 40.0]
        # class 1 확률 반환
        assert out[0] == pytest.approx(0.8)

    def test_works_without_scaler(self):
        """scaler 가 None 이면 스케일 없이 결합만 적용."""
        xgb = MagicMock()
        xgb.predict_proba.return_value = np.array([[0.1, 0.9]])
        out = predict_fraud_probs(np.array([[5.0]]), np.array([[1.0]]), xgb, None)
        proba_input = xgb.predict_proba.call_args[0][0]
        assert list(proba_input[0]) == [1.0, 5.0]   # [원본, 임베딩]
        assert out[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. 통합 회귀 테스트 (실제 아티팩트 필요 — 없으면 skip)
# ---------------------------------------------------------------------------

_GRAPH = ROOT / "model" / "processed_graph_data.pt"
_GNN   = ROOT / "gnn_model.pth"
_XGB   = ROOT / "fraud_model.pkl"
_ARTIFACTS_PRESENT = _GRAPH.exists() and _GNN.exists() and _XGB.exists()


@pytest.mark.skipif(not _ARTIFACTS_PRESENT, reason="모델 아티팩트 없음 (.pt/.pth/.pkl)")
class TestPipelineAUCRegression:
    @pytest.fixture(scope="class")
    def scored(self):
        import torch
        import joblib
        from models.embedding_extractor import EmbeddingExtractor

        g = torch.load(str(_GRAPH), map_location="cpu")
        extractor = EmbeddingExtractor(
            in_channels=g["x"].shape[1], hidden_channels=128, embed_dim=64
        )
        extractor.load_state_dict(torch.load(str(_GNN), map_location="cpu"))
        extractor.eval()
        with torch.no_grad():
            embs = extractor(g["x"], g["edge_index"]).numpy()

        data = joblib.load(str(_XGB))
        probs = predict_fraud_probs(
            embs, g["x"].numpy(), data["xgb_model"], data.get("scaler")
        )
        return probs, g["y"].numpy(), g["test_mask"].numpy().astype(bool)

    def test_test_set_auc_above_0_9(self, scored):
        """올바른 파이프라인은 test AUC > 0.9 (과거 버그 시 0.34)."""
        from sklearn.metrics import roc_auc_score
        probs, y, tm = scored
        auc = roc_auc_score(y[tm], probs[tm])
        assert auc > 0.9, f"test AUC={auc:.3f} — 추론 파이프라인 회귀 의심"

    def test_known_fraud_node_scores_high(self, scored):
        """실제 사기 노드(61366)는 높은 확률을 받아야 한다."""
        probs, y, _ = scored
        assert y[61366] == 1
        assert probs[61366] > 0.7

    def test_scaler_is_bundled(self):
        """fraud_model.pkl 에 추론용 StandardScaler 가 포함돼야 한다."""
        import joblib
        data = joblib.load(str(_XGB))
        assert data.get("scaler") is not None
