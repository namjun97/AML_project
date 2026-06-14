import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EmbeddingExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()

        # --- GraphSAGE 레이어 3층 ---
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, embed_dim)

        # --- 정규화 ---
        self.dropout = nn.Dropout(p=0.3)

        # --- 이진 분류 헤드 (0: 정상, 1: 사기) ---
        self.classifier = nn.Linear(embed_dim, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # 마지막 레이어는 활성화 없이 임베딩만 반환
        x = self.conv3(x, edge_index)
        return x

    # ------------------------------------------------------------------
    def classify(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.forward(x, edge_index)
        return self.classifier(emb)


# ======================================================================
# XGBoost 하이브리드 추론 — 단일 소스 (app / resource_loader / neo4j_loader 공유)
#
# 학습 파이프라인(aml_model_comparison.ipynb)과 정확히 일치시킨다:
#   CELL 24: hybrid = np.hstack([origin_features(10), node_embeddings(64)])  # 원본 먼저!
#   CELL 30: scaler.fit(train) -> scaler.transform(전체)  # 74-dim StandardScaler
#   CELL 32: xgb_model.fit(scaled, y)
#
# [과거 버그] 서빙 3경로가 모두 hstack([emb, orig]) 로 순서를 뒤집고 스케일러를
# 생략 -> XGBoost 입력이 학습과 어긋나 fraud_prob 의 AUC 가 0.34(무작위 이하)로
# 무의미했음. 아래 헬퍼로 통일해 동일 버그 재발을 차단한다. (회귀 테스트로 AUC>0.9 검증)
# ======================================================================

def build_hybrid_features(orig_feats: np.ndarray, embs: np.ndarray) -> np.ndarray:
    """학습과 동일한 [원본 집계 10, GNN 임베딩 64] 순서로 결합한다."""
    orig_feats = np.asarray(orig_feats, dtype=np.float32)
    embs       = np.asarray(embs,       dtype=np.float32)
    return np.hstack([orig_feats, embs]).astype(np.float32)


def apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """확률에 temperature scaling 적용 (과신 완화 보정).

    predict_proba 출력에서 logit 을 역산해 T 로 나눈 뒤 다시 sigmoid.
    (XGBoost output_margin 은 base_score 처리 차이로 predict_proba 와 어긋나므로
     predict_proba 에서 직접 역산한다.) T>1 이면 과신 완화, T==1 이면 무변환.
    단조 변환이라 순위(AUC)는 보존된다. T 는 검증셋 NLL 최소화로 적합.
    """
    if temperature is None or abs(float(temperature) - 1.0) < 1e-6:
        return probs
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / float(temperature)))


def predict_fraud_probs(
    embs: np.ndarray,
    orig_feats: np.ndarray,
    xgb_model,
    scaler,
    temperature: float = 1.0,
) -> np.ndarray:
    """GNN 임베딩 + 원본 피처 -> 사기 확률(class 1). 학습 전처리와 동일.

    Args:
        embs:       (N, 64) GNN 임베딩
        orig_feats: (N, 노드피처) 원본 집계 피처 (graph_dict["x"], 이미 1차 스케일됨)
        xgb_model:  학습된 XGBClassifier
        scaler:     fraud_model.pkl 의 StandardScaler (없으면 생략)
        temperature: 과신 완화 보정 계수 (fraud_model.pkl 의 'temperature', 기본 1.0=무변환)

    Returns:
        (N,) 사기 확률 배열
    """
    X = build_hybrid_features(orig_feats, embs)
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    probs = xgb_model.predict_proba(X)[:, 1]
    return apply_temperature(probs, temperature)


def hybrid_feature_names(all_feature_names: list) -> list:
    """저장된 all_feature_names 를 실제 학습 피처 순서([원본10, 임베딩64])로 재정렬.

    fraud_model.pkl 의 all_feature_names 는 [GNN_Emb*64, 원본*10] 으로 저장돼 있어
    실제 학습 행렬([원본, 임베딩])과 어긋난다. SHAP/워터폴/SAR 의 피처 라벨이
    컬럼과 일치하도록 [원본, 임베딩] 순서로 되돌린다.
    """
    emb  = [n for n in all_feature_names if str(n).startswith("GNN_Emb")]
    orig = [n for n in all_feature_names if not str(n).startswith("GNN_Emb")]
    return orig + emb
