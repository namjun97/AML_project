"""
analysis/network_visualizer.py
--------------------------------
pyvis 기반 자금 흐름 네트워크 시각화 모듈.

[두 가지 관계 레이어]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 실거래 엣지 (Transaction Layer)
   - edge_index에서 추출한 실제 송금 거래
   - 방향성 화살표 (자금 흐름 방향)
   - 선 굵기 = 거래 금액 비례
   - 색상: 사기 관련=빨강, 정상=회색

② 패턴 유사 엣지 (Pattern Similarity Layer)
   - GNN 임베딩 공간에서 코사인 유사도 KNN
   - 직접 거래 관계가 없어도 비슷한 행동 패턴의 계좌를 탐지
   - 점선(dashes=True), 연보라색
   - AML 수사에서 "동일 세탁 링" 소속 계좌 탐지에 유용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[그래프 희소성 문제 해결]
PaySim 그래프의 평균 degree ≈ 0.52 (111,427 노드 / 58,213 엣지).
실거래 BFS만으로는 대부분의 노드가 1~3개 이웃만 탐색됨.
→ GNN 임베딩 KNN으로 유사 패턴 계좌를 추가하여 의미 있는 클러스터 시각화.

[거래 금액 표시]
edge_attr[:,0] = log_amount = np.log1p(amount)
역변환: original_amount = np.expm1(log_amount)
모든 금액 표시는 원래 금액(₩) 기준으로 출력.

[edge_attr 컬럼 인덱스 — aml_project.ipynb EDGE_FEATURE_COLS 기준]
  index 0 : log_amount        (로그 변환 거래 금액)
  index 1 : type_encoded      (TRANSFER=0, CASH_OUT=1)
  index 2 : balance_diff_orig (송금자 잔액 변화량)
  index 3 : balance_diff_dest (수취자 잔액 변화량)
  index 4 : balance_mismatch  (잔액 불일치)
  index 5 : hour_of_day_norm  [수정] 하루 중 거래 시간대 (step%24 / 23, 0~1)
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from pyvis.network import Network


# ======================================================================
# 상수
# ======================================================================
_EDGE_COLS = ["log_amount", "type_encoded", "balance_diff_orig",
              "balance_diff_dest", "balance_mismatch", "hour_of_day_norm"]
_TYPE_MAP  = {0: "TRANSFER", 1: "CASH_OUT"}

# 노드 색상
_COLOR_CENTER = "#FFD700"
_COLOR_CENTER_BORDER = "#FF8C00"
_COLOR_FRAUD  = "#EF5350"
_COLOR_FRAUD_BORDER = "#B71C1C"
_COLOR_NORMAL = "#42A5F5"
_COLOR_NORMAL_BORDER = "#1565C0"

# 엣지 색상
_COLOR_EDGE_FRAUD  = "#FF5252"   # 실거래 · 사기 관련
_COLOR_EDGE_NORMAL = "#78909C"   # 실거래 · 정상
_COLOR_EDGE_SIM    = "#CE93D8"   # 패턴 유사 (연보라 점선)

_BG_COLOR = "#0e1117"


# ======================================================================
# 헬퍼 함수
# ======================================================================

def _expm1_amount(log_amount: float) -> float:
    """log_amount(np.log1p 변환값)를 원래 금액으로 역변환합니다."""
    return float(np.expm1(log_amount))


def _format_amount(log_amount: float) -> str:
    """거래 금액을 읽기 좋은 문자열로 포맷합니다."""
    amount = _expm1_amount(log_amount)
    if amount >= 1_000_000:
        return f"₩{amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"₩{amount:,.0f}"
    else:
        return f"₩{amount:.2f}"


def _find_transaction_neighbors(
    seed_nodes: set[int],
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    max_total: int,
    visited: set[int],
    hop: int = 2,
) -> set[int]:
    """
    BFS로 실거래 이웃 노드를 탐색합니다.

    Args:
        seed_nodes : 탐색 시작 노드 집합
        src_arr    : edge_index[0] (출발 노드 배열)
        dst_arr    : edge_index[1] (도착 노드 배열)
        max_total  : 최대 포함 노드 수
        visited    : 이미 포함된 노드 (수정됨)
        hop        : 탐색 깊이

    Returns:
        set[int]: 탐색된 이웃 노드 집합 (visited 포함)
    """
    frontier = set(seed_nodes)
    for _ in range(hop):
        next_frontier: set[int] = set()
        for node in frontier:
            out_nodes = dst_arr[src_arr == node].tolist()
            in_nodes  = src_arr[dst_arr == node].tolist()
            next_frontier.update(out_nodes + in_nodes)
        next_frontier -= visited
        remaining = max_total - len(visited)
        if remaining <= 0:
            break
        if len(next_frontier) > remaining:
            next_frontier = set(random.sample(list(next_frontier), remaining))
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def _find_embedding_neighbors(
    center_idx: int,
    all_embs: np.ndarray,
    exclude_ids: set[int],
    y_labels: np.ndarray,
    k: int = 30,
    fraud_boost: float = 0.08,
) -> list[tuple[int, float]]:
    """
    GNN 임베딩 공간에서 center_idx와 코사인 유사도가 높은 노드 k개를 반환합니다.

    사기 노드에 fraud_boost를 가산해 우선 탐색되도록 합니다.

    Args:
        center_idx  : 기준 노드 인덱스
        all_embs    : 전체 노드 임베딩 [N, D]
        exclude_ids : 제외할 노드 인덱스 집합
        y_labels    : 노드 라벨 배열
        k           : 반환할 이웃 수
        fraud_boost : 사기 노드 유사도 가산점

    Returns:
        list[tuple[int, float]]: (노드 인덱스, 유사도) 리스트
    """
    center_emb  = all_embs[center_idx]
    center_norm = center_emb / (np.linalg.norm(center_emb) + 1e-8)

    # 전체 코사인 유사도 계산 (행렬 연산으로 고속 처리)
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
    sims  = (all_embs / norms) @ center_norm  # [N]

    # 제외 노드 마스킹
    exclude_list = list(exclude_ids)
    if exclude_list:
        sims[exclude_list] = -2.0

    # 사기 노드 가산점 (수사 관련성 높은 노드 우선 탐색)
    if fraud_boost > 0:
        fraud_mask = (y_labels == 1)
        sims[fraud_mask] += fraud_boost

    # 상위 k개 추출
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return [(int(idx), float(sims[idx])) for idx in top_k_idx]


# ======================================================================
# 메인 시각화 함수
# ======================================================================

def build_fund_flow_network(
    selected_idx: int,
    graph_dict: dict[str, Any],
    risk_df: pd.DataFrame,
    all_embs: np.ndarray | None = None,
    hop: int = 2,
    max_transaction_nodes: int = 30,
    max_similarity_nodes: int = 30,
    height: str = "560px",
) -> str:
    """
    선택된 노드 중심의 자금 흐름 네트워크를 pyvis HTML로 반환합니다.

    실거래 엣지(실선)와 GNN 패턴 유사 엣지(점선) 두 레이어를 함께 시각화합니다.

    Args:
        selected_idx          : 분석 대상 노드 인덱스
        graph_dict            : load_resources() 반환 그래프 딕셔너리
        risk_df               : 위험도 DataFrame (node_idx, fraud_prob)
        all_embs              : 전체 노드 GNN 임베딩 [N, D] (None이면 KNN 비활성)
        hop                   : 실거래 BFS 탐색 깊이 (default: 2)
        max_transaction_nodes : 실거래 BFS 최대 노드 수 (default: 30)
        max_similarity_nodes  : KNN 유사 패턴 최대 노드 수 (default: 30)
        height                : 시각화 높이 (default: "560px")

    Returns:
        str: pyvis HTML 문자열
    """
    edge_index = graph_dict["edge_index"].numpy()
    edge_attr  = graph_dict["edge_attr"].numpy()   # SAML: [log_amount, cross_border, currency_mismatch]
    y_labels   = graph_dict["y"].numpy()
    # 세탁 거래 플래그 (없으면 노드 라벨 기반으로 폴백)
    _eil = graph_dict.get("edge_is_laundering")
    edge_is_l = _eil.numpy() if _eil is not None else None
    src_arr    = edge_index[0]
    dst_arr    = edge_index[1]

    prob_map: dict[int, float] = dict(
        zip(risk_df["node_idx"].tolist(), risk_df["fraud_prob"].tolist())
    )

    # ── 레이어 1: 실거래 이웃 탐색 (BFS) ─────────────────────────────
    tx_nodes: set[int] = {selected_idx}
    tx_nodes = _find_transaction_neighbors(
        seed_nodes={selected_idx},
        src_arr=src_arr,
        dst_arr=dst_arr,
        max_total=max_transaction_nodes,
        visited=tx_nodes,
        hop=hop,
    )

    # ── 레이어 2: GNN 임베딩 유사 패턴 이웃 탐색 (KNN) ───────────────
    sim_nodes: set[int] = set()
    sim_edges: list[tuple[int, float]] = []   # (node_idx, similarity_score)

    if all_embs is not None and len(all_embs) > selected_idx:
        knn_results = _find_embedding_neighbors(
            center_idx  = selected_idx,
            all_embs    = all_embs,
            exclude_ids = tx_nodes,          # 이미 포함된 실거래 노드 제외
            y_labels    = y_labels,
            k           = max_similarity_nodes,
            fraud_boost = 0.08,
        )
        for node_idx, sim_score in knn_results:
            sim_nodes.add(node_idx)
            sim_edges.append((node_idx, sim_score))

    all_nodes = tx_nodes | sim_nodes

    # ── pyvis Network 초기화 ──────────────────────────────────────────
    net = Network(
        height=height,
        width="100%",
        bgcolor=_BG_COLOR,
        font_color="white",
        directed=True,
        notebook=False,
    )
    net.toggle_physics(True)

    # ── 노드 추가 ────────────────────────────────────────────────────
    for node in all_nodes:
        label_val = int(y_labels[node]) if node < len(y_labels) else 0
        prob      = prob_map.get(node)
        is_center = (node == selected_idx)
        is_sim    = (node in sim_nodes and node not in tx_nodes)

        if is_center:
            color  = {"background": _COLOR_CENTER,
                      "border": _COLOR_CENTER_BORDER,
                      "highlight": {"background": "#FFF176", "border": _COLOR_CENTER_BORDER}}
            size, border_w = 30, 3
        elif label_val == 1:
            color  = {"background": _COLOR_FRAUD,
                      "border": _COLOR_FRAUD_BORDER,
                      "highlight": {"background": "#FF8A80", "border": _COLOR_FRAUD_BORDER}}
            size, border_w = 18, 2
        else:
            color  = {"background": _COLOR_NORMAL,
                      "border": _COLOR_NORMAL_BORDER,
                      "highlight": {"background": "#90CAF9", "border": _COLOR_NORMAL_BORDER}}
            size, border_w = 12, 1

        # 패턴 유사 노드는 테두리를 보라색으로 구분
        if is_sim:
            color["border"] = "#CE93D8"
            size = max(size, 13)

        prob_str  = f"{prob:.1%}" if prob is not None else "N/A"
        layer_tag = "⭐ 분석 대상" if is_center else ("🔮 패턴 유사" if is_sim else "📊 거래 연결")
        label_str = "🔴 사기" if label_val == 1 else "🔵 정상"

        title = (
            f"<b>노드 ID: {node}</b><br>"
            f"분류: {layer_tag}<br>"
            f"실제 라벨: {label_str}<br>"
            f"사기 의심 확률: {prob_str}"
        )

        net.add_node(
            n_id        = node,
            label       = str(node),
            color       = color,
            size        = size,
            title       = title,
            borderWidth = border_w,
            font        = {"size": 11, "color": "white"},
        )

    # ── 실거래 엣지 추가 ─────────────────────────────────────────────
    node_set  = all_nodes
    src_in    = np.isin(src_arr, list(node_set))
    dst_in    = np.isin(dst_arr, list(node_set))
    edge_mask = src_in & dst_in
    v_src     = src_arr[edge_mask]
    v_dst     = dst_arr[edge_mask]
    v_attr    = edge_attr[edge_mask]
    v_laund   = edge_is_l[edge_mask] if edge_is_l is not None else None

    for i, (s, d, attr) in enumerate(zip(v_src, v_dst, v_attr)):
        log_amount  = float(attr[0])
        cross       = len(attr) > 1 and float(attr[1]) >= 0.5     # 국가 간 거래
        curr_mis    = len(attr) > 2 and float(attr[2]) >= 0.5     # 송·수취 통화 불일치
        orig_amount = _expm1_amount(log_amount)

        # 세탁 거래 플래그: 엣지 단위 라벨 우선, 없으면 노드 라벨 기반 폴백
        if v_laund is not None:
            is_fraud_edge = bool(int(v_laund[i]))
        else:
            is_fraud_edge = (
                (int(y_labels[s]) == 1 if s < len(y_labels) else False) or
                (int(y_labels[d]) == 1 if d < len(y_labels) else False)
            )
        edge_color = _COLOR_EDGE_FRAUD if is_fraud_edge else _COLOR_EDGE_NORMAL
        edge_width = max(1.5, min(8.0, log_amount * 0.7))

        flags = []
        if is_fraud_edge:
            flags.append("🔴 자금세탁 확정 거래")
        if cross:
            flags.append("국가 간 거래(역외)")
        if curr_mis:
            flags.append("송·수취 통화 불일치(환치기)")
        flag_str = ("<br>" + " · ".join(flags)) if flags else ""
        title = (
            f"<b>거래 금액: ₩{orig_amount:,.0f}</b>"
            f"{flag_str}"
        )
        net.add_edge(
            source = int(s),
            to     = int(d),
            width  = edge_width,
            color  = {"color": edge_color, "highlight": "#FFFFFF"},
            title  = title,
            arrows = "to",
        )

    # ── 패턴 유사 엣지 추가 (선택 노드 ↔ KNN 유사 노드, 점선) ────────
    for sim_node, sim_score in sim_edges:
        if sim_node not in all_nodes:
            continue
        sim_label = int(y_labels[sim_node]) if sim_node < len(y_labels) else 0
        sim_prob  = prob_map.get(sim_node)
        sim_prob_str = f"{sim_prob:.1%}" if sim_prob is not None else "N/A"

        title = (
            f"<b>패턴 유사도 엣지</b><br>"
            f"유사도 (코사인): {sim_score:.4f}<br>"
            f"대상 노드: {sim_node}<br>"
            f"라벨: {'🔴 사기' if sim_label == 1 else '🔵 정상'}<br>"
            f"사기 확률: {sim_prob_str}<br>"
            f"<i>GNN 임베딩 공간에서 유사한 거래 패턴</i>"
        )
        net.add_edge(
            source = selected_idx,
            to     = sim_node,
            width  = max(1.0, sim_score * 3),
            color  = {"color": _COLOR_EDGE_SIM, "highlight": "#E040FB"},
            title  = title,
            arrows = "",           # 방향 없음 (유사도는 무방향)
            dashes = True,
        )

    # ── 물리 엔진 옵션 ───────────────────────────────────────────────
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "shadow": { "enabled": true, "size": 6 }
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "shadow": false,
        "selectionWidth": 2
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.1,
          "avoidOverlap": 0.2
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 200,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "navigationButtons": true,
        "zoomSpeed": 1
      }
    }
    """)

    return net.generate_html()


# ======================================================================
# 네트워크 통계 요약
# ======================================================================

def get_network_stats(
    selected_idx: int,
    graph_dict: dict[str, Any],
    risk_df: pd.DataFrame,
    all_embs: np.ndarray | None = None,
    hop: int = 2,
    max_transaction_nodes: int = 30,
    max_similarity_nodes: int = 30,
) -> dict:
    """
    네트워크 시각화에 포함될 노드/엣지 통계를 반환합니다.

    거래 금액은 log 역변환하여 원래 금액(₩) 기준으로 반환합니다.

    Returns:
        dict: {
            tx_nodes, sim_nodes, total_nodes,
            fraud_nodes, normal_nodes,
            tx_edges, fraud_edges,
            max_amount_original (₩, int),
            avg_amount_original (₩, int),
        }
    """
    edge_index = graph_dict["edge_index"].numpy()
    edge_attr  = graph_dict["edge_attr"].numpy()
    y_labels   = graph_dict["y"].numpy()
    src_arr    = edge_index[0]
    dst_arr    = edge_index[1]

    # 실거래 이웃 (BFS)
    tx_nodes: set[int] = {selected_idx}
    tx_nodes = _find_transaction_neighbors(
        seed_nodes={selected_idx},
        src_arr=src_arr, dst_arr=dst_arr,
        max_total=max_transaction_nodes,
        visited=tx_nodes, hop=hop,
    )

    # 유사 패턴 이웃 (KNN)
    sim_nodes: set[int] = set()
    if all_embs is not None and len(all_embs) > selected_idx:
        knn = _find_embedding_neighbors(
            center_idx=selected_idx, all_embs=all_embs,
            exclude_ids=tx_nodes, y_labels=y_labels,
            k=max_similarity_nodes, fraud_boost=0.08,
        )
        sim_nodes = {idx for idx, _ in knn}

    all_nodes = tx_nodes | sim_nodes
    node_set  = all_nodes

    # 엣지 필터
    src_in    = np.isin(src_arr, list(node_set))
    dst_in    = np.isin(dst_arr, list(node_set))
    edge_mask = src_in & dst_in
    v_src     = src_arr[edge_mask]
    v_dst     = dst_arr[edge_mask]
    v_attr    = edge_attr[edge_mask]

    fraud_nodes = sum(1 for n in all_nodes if n < len(y_labels) and y_labels[n] == 1)
    fraud_edges = sum(
        1 for s, d in zip(v_src, v_dst)
        if (s < len(y_labels) and y_labels[s] == 1) or
           (d < len(y_labels) and y_labels[d] == 1)
    )

    log_amounts = v_attr[:, 0] if len(v_attr) > 0 else np.array([0.0])
    max_log   = float(np.max(log_amounts))
    avg_log   = float(np.mean(log_amounts))

    return {
        "tx_nodes":           len(tx_nodes),
        "sim_nodes":          len(sim_nodes),
        "total_nodes":        len(all_nodes),
        "fraud_nodes":        fraud_nodes,
        "normal_nodes":       len(all_nodes) - fraud_nodes,
        "tx_edges":           int(edge_mask.sum()),
        "fraud_edges":        fraud_edges,
        # log 역변환 → 원래 금액
        "max_amount_original": int(_expm1_amount(max_log)),
        "avg_amount_original": int(_expm1_amount(avg_log)),
    }
