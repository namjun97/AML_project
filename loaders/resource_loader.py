import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st

from models.embedding_extractor import EmbeddingExtractor, predict_fraud_probs

# 1. 모델 및 데이터 로드

@st.cache_resource
def load_resources(
    graph_path: str = "model/saml_graph.pt",
    gnn_path: str = "model/saml_gnn.pth",
    xgb_path: str = "model/saml_fraud_model.pkl",
) -> tuple:
    # 1-1. 그래프 데이터 로드
    graph_dict = torch.load(graph_path, map_location="cpu")

    # 1-2. GNN 모델 초기화 및 가중치 로드
    in_dim = graph_dict["x"].shape[1]
    extractor = EmbeddingExtractor(
        in_channels=in_dim,
        hidden_channels=128,
        embed_dim=64,
    )
    extractor.load_state_dict(torch.load(gnn_path, map_location="cpu"))
    extractor.eval()

    # 1-3. XGBoost 모델 + 피처명 + 스케일러 로드
    #   scaler: 학습 시 74-dim 하이브리드 피처에 적용한 StandardScaler.
    #   추론에서 반드시 동일 적용해야 한다 (predict_fraud_probs 참고).
    xgb_data = joblib.load(xgb_path)
    xgb_model = xgb_data["xgb_model"]
    all_feature_names = xgb_data["all_feature_names"]
    scaler = xgb_data.get("scaler")

    return graph_dict, extractor, xgb_model, all_feature_names, scaler


# 2. 전체 노드 임베딩 사전 계산

@st.cache_data
def get_all_embeddings_cached(
    _extractor: EmbeddingExtractor,
    _x: torch.Tensor,
    _edge_index: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        embeddings = _extractor(_x, _edge_index)
    return embeddings.numpy()


# 3. 위험 순위 DataFrame 생성

@st.cache_data
def get_sorted_test_nodes(
    _all_embs: np.ndarray,
    _graph_dict: dict,
    _xgb_model,
    _all_feature_names: list,
    _scaler=None,
) -> pd.DataFrame:
    test_mask = _graph_dict["test_mask"]
    test_idx = np.where(test_mask.numpy())[0]

    # 임베딩 (미리 계산된 값에서 인덱싱)
    test_embeddings = _all_embs[test_idx]

    # 원본 노드 피처 전체 (graph_dict["x"] 가 곧 노드 피처 — SAML 11개)
    test_orig = _graph_dict["x"][test_idx].numpy()

    # 사기 확률 예측 — 학습과 동일한 [원본, 임베딩] 순서 + 스케일러 (단일 헬퍼)
    probs = predict_fraud_probs(test_embeddings, test_orig, _xgb_model, _scaler)

    risk_df = pd.DataFrame(
        {
            "node_idx": test_idx,
            "fraud_prob": probs,
            "key_feature": _graph_dict["x"][test_idx, 0].numpy(),
        }
    ).sort_values(by="fraud_prob", ascending=False)

    return risk_df
