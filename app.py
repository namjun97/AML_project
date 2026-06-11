from dotenv import load_dotenv
load_dotenv()

import requests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from analysis.network_visualizer import build_fund_flow_network, get_network_stats
from analysis.shap_analyzer import ShapAnalyzer
from loaders.resource_loader import (
    get_all_embeddings_cached,
    get_sorted_test_nodes,
    load_resources,
)
from loaders.knowledge_loader import init_knowledge_resources
from reporters.ai_report_generator import (
    _humanize_feature_value,
    build_sar_payload,
)
from reporters.context_builder import ContextBuilder
from reporters.pdf_report_generator import create_pdf_report
from reporters.report_runner import ReportRunner
from reporters.sar_graph import build_sar_graph


# 0. 페이지 설정

st.set_page_config(page_title="자금 세탁 경로 탐지 시스템", layout="wide")
st.title("🛡️ 자금 세탁 거점 분석 대시보드")


# 1. 모델·그래프 데이터 로드 (캐싱)

graph_dict, extractor, xgb_model, all_feature_names = load_resources()
all_embs_cached = get_all_embeddings_cached(
    extractor, graph_dict["x"], graph_dict["edge_index"]
)
risk_df = get_sorted_test_nodes(
    all_embs_cached, graph_dict, xgb_model, all_feature_names
)
high_risk_indices = risk_df["node_idx"].tolist()


# 2. 지식 리소스 초기화 (텍스트 RAG + GraphRAG)

kr = init_knowledge_resources()

# 사이드바 상태 표시
if kr.rag_error:
    st.sidebar.warning(
        f"⚠️ 텍스트 RAG 초기화 실패: {kr.rag_error}\n\n"
        "`ollama pull nomic-embed-text` 실행 후 앱을 재시작하세요."
    )
if kr.graph_retriever is not None and not kr.graph_retriever.connect_attempted:
    if kr.graph_available:
        st.sidebar.info("🗄️ Neo4j GraphRAG — 첫 SAR 생성 시 연결 (지연 연결)")
    else:
        st.sidebar.info("ℹ️ Neo4j 접속 정보 없음 — 텍스트 RAG만 사용")
elif kr.graph_available:
    st.sidebar.success("🗄️ Neo4j GraphRAG 연결됨")
else:
    st.sidebar.info("ℹ️ Neo4j 미연결 — 텍스트 RAG만 사용")

# 공유 객체 조립
context_builder = ContextBuilder(
    knowledge_base  = kr.knowledge_base,
    graph_retriever = kr.graph_retriever,
    rag_available   = kr.rag_available,
    graph_available = kr.graph_available,
)
report_runner = ReportRunner(context_builder)


@st.cache_resource
def get_sar_graph():
    return build_sar_graph(context_builder, report_runner)


sar_graph = get_sar_graph()

# ======================================================================
# 3. 세션 상태 초기화
# ======================================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "ai_report_content" not in st.session_state:
    st.session_state.ai_report_content = None
if "current_node" not in st.session_state:
    st.session_state.current_node = None
if "rag_context_used" not in st.session_state:
    st.session_state.rag_context_used = ""
if "graph_context_used" not in st.session_state:
    st.session_state.graph_context_used = ""

# ======================================================================
# 4. 사이드바 — 노드 선택
# ======================================================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.info("💡 아래 리스트는 모델이 판단한 **사기 의심 확률이 높은 순**으로 정렬되어 있습니다.")

    def format_node_label(idx: int) -> str:
        row = risk_df[risk_df["node_idx"] == idx].iloc[0]
        return f"ID: {idx} (위험도: {row['fraud_prob']:.1%})"

    selected_idx = st.selectbox(
        "분석할 거래(노드) 선택",
        high_risk_indices,
        format_func=format_node_label,
    )

    # 노드가 바뀌면 이전 분석 결과 전체 초기화
    if selected_idx != st.session_state.current_node:
        st.session_state.analysis_done      = False
        st.session_state.ai_report_content  = None
        st.session_state.current_node       = selected_idx
        st.session_state.rag_context_used   = ""
        st.session_state.graph_context_used = ""

    analyze_btn = st.button("🔍 상세 분석 실행")

if analyze_btn:
    st.session_state.analysis_done = True

# ======================================================================
# 5. 메인 분석 영역
# ======================================================================
if not st.session_state.analysis_done:
    st.write(
        "👈 왼쪽 사이드바에서 노드를 선택하고 분석 버튼을 눌러주세요. "
        "(리스트 상단에 위험 계좌가 배치되어 있습니다)"
    )
    st.stop()

with st.spinner("분석 중......"):
    # 5-1. 임베딩 인덱싱 + XGBoost 예측
    node_emb  = all_embs_cached[selected_idx].reshape(1, -1)
    node_orig = graph_dict["x"][selected_idx][-10:].reshape(1, -1).numpy()
    X_input   = np.hstack([node_emb, node_orig])
    prob      = xgb_model.predict_proba(X_input)[0][1]

    # 5-2. SHAP 분석
    analyzer = ShapAnalyzer(xgb_model)
    analyzer.build_explainer(X_input)
    if analyzer.is_kernel:
        st.warning("TreeExplainer 초기화 실패 — KernelExplainer로 대체 실행 중입니다.")
    shap_values = analyzer.compute_shap_values(X_input)

    # 5-3. 메트릭 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("선택된 노드", selected_idx)
    c2.metric("자금 세탁 통로 의심 점수", f"{prob:.2%}")
    if prob > 0.5:
        c3.error("🚨 고위험 계좌(자금 세탁 의심 계좌)")
        risk_level = "위험(Danger)"
    else:
        c3.success("✅ 정상 거래 계좌")
        risk_level = "안전(Safe)"

    st.divider()

    # 5-4. SHAP 시각화 + 피처 테이블
    st.write("### 📊 판단 근거 분석")
    top_feature = analyzer.most_influential_feature(shap_values, all_feature_names)
    st.info(f"#### 💡 분석 핵심 요약: 가장 큰 영향을 준 요인은 **'{top_feature}'** 입니다.")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("#### 1. 피처별 기여도 (영향력 TOP 10)")
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.plots._waterfall.waterfall_legacy(
            analyzer.expected_value,
            shap_values[0],
            feature_names=all_feature_names,
            max_display=10,
            show=False,
        )
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.write("#### 2. 원본 피처 데이터 정보")
        orig_feature_names = all_feature_names[-10:]
        orig_df = pd.DataFrame(node_orig, columns=orig_feature_names)

        # z-score 수치 + 수준 레이블을 한 열에 함께 표시
        readable_rows = []
        for feat in orig_feature_names:
            z_val = float(orig_df[feat].iloc[0])
            human = _humanize_feature_value(z_val, feat)
            readable_rows.append({
                "피처명":       feat,
                "수치 (z점수)": human["z점수"],
                "수준":         f"{human['수준']}  ({human['z점수']:+.4f})",
                "의미":         human["설명"],
            })
        readable_df = pd.DataFrame(readable_rows).set_index("피처명")
        st.dataframe(readable_df, use_container_width=True)

    st.success(
        f"""
        **💡 분석 결과 요약:**
        - 현재 이 거래는 **{prob:.2%}**의 확률로 사기일 가능성이 있습니다.
        - 오른쪽(분홍색/양수) 막대는 사기 확률을 **증가**시키는 요인입니다.
        - 왼쪽(하늘색/음수) 막대는 사기 확률을 **감소**시키는 요인입니다.
        - GNN_Emb로 시작하는 항목은 인접한 노드(거래처)와의 관계에서 도출된 특징입니다.
        """
    )

    # 5-5. 자금 흐름 네트워크 시각화 (pyvis)
    st.divider()
    st.write("### 🕸️ 자금 흐름 네트워크 시각화")
    st.caption(
        "**실선**: 실제 송금 거래 (방향 = 자금 흐름) &nbsp;·&nbsp; "
        "**점선**: GNN 임베딩 유사 패턴 계좌 (동일 세탁 링 탐지) &nbsp;|&nbsp; "
        "🟡 분석 대상 · 🔴 사기 · 🔵 정상 · 보라 테두리 = 패턴 유사 계좌 &nbsp;|&nbsp; "
        "마우스 오버로 상세 정보 확인"
    )

    with st.spinner("네트워크 그래프 생성 중..."):
        try:
            net_stats = get_network_stats(
                selected_idx=selected_idx,
                graph_dict=graph_dict,
                risk_df=risk_df,
                all_embs=all_embs_cached,
                hop=2,
                max_transaction_nodes=30,
                max_similarity_nodes=30,
            )
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("총 노드 수",    f"{net_stats['total_nodes']}개")
            sc2.metric("실거래 노드",   f"{net_stats['tx_nodes']}개",
                        help="실제 거래로 연결된 계좌")
            sc3.metric("패턴 유사 노드", f"{net_stats['sim_nodes']}개",
                        help="GNN 임베딩 공간에서 유사한 행동 패턴을 가진 계좌")
            sc4.metric("사기 노드 수",  f"{net_stats['fraud_nodes']}개",
                        delta=f"정상 {net_stats['normal_nodes']}개",
                        delta_color="inverse")
            sc5.metric("최대 거래 금액",
                        f"₩{net_stats['max_amount_original']:,}",
                        help="그래프 내 실거래 최대 금액 (원래 값)")

            net_html = build_fund_flow_network(
                selected_idx=selected_idx,
                graph_dict=graph_dict,
                risk_df=risk_df,
                all_embs=all_embs_cached,
                hop=2,
                max_transaction_nodes=30,
                max_similarity_nodes=30,
                height="560px",
            )
            components.html(net_html, height=590, scrolling=False)

        except Exception as _net_err:
            st.warning(f"네트워크 시각화 생성 실패: {_net_err}")

    # 5-6. SAR 페이로드 조립
    st.divider()
    st.write("### 📝 SAR 보고서 작성을 위한 데이터 추출")
    orig_df_dict = orig_df.to_dict(orient="records")[0]
    sar_payload, sar_json_str = build_sar_payload(
        selected_idx      = selected_idx,
        prob              = prob,
        shap_values       = shap_values,
        X_input           = X_input,
        all_feature_names = all_feature_names,
        orig_df_dict      = orig_df_dict,
    )

    # 5-7. AI SAR 보고서 생성
    st.divider()
    st.subheader("AI 자동 생성 의심거래보고서(SAR)")

    if st.button("🚀 AI 전문 보고서 생성(Llama 3.1)"):

        # ── ① 컨텍스트 조회 (ReportRunner → ContextBuilder) ─────────
        with st.spinner("🔍 RAG·GraphRAG 컨텍스트 조회 중..."):
            rag_context, graph_context = report_runner.build_contexts(
                sar_payload, selected_idx
            )

        # ── ② 스트리밍 보고서 생성 ──────────────────────────────────
        progress_placeholder = st.empty()
        raw_text  = ""
        error_msg = ""

        try:
            for token in report_runner.stream(sar_json_str, rag_context, graph_context):
                raw_text += token
                progress_placeholder.info(
                    f"⏳ AI 분석 중... ({len(raw_text)}자 생성됨)\n\n"
                    f"완료 후 통일된 보고서 양식으로 자동 변환됩니다."
                )
        except requests.exceptions.ConnectionError:
            error_msg = "❌ Ollama 서버에 연결할 수 없습니다. localhost:11434 가 실행 중인지 확인하세요."
        except Exception as _exc:
            error_msg = f"❌ 보고서 생성 오류: {_exc}"

        progress_placeholder.empty()

        if error_msg:
            st.error(error_msg)
            final_report = error_msg
        else:
            # ── ③ RAG 에코 제거 + 양식 조립 (ReportRunner) ──────────
            final_report = report_runner.finalize(raw_text, sar_payload, graph_context)

            st.markdown(
                "<div style='background:#f0f2f6;padding:20px;border-radius:8px;"
                "border-left:5px solid #ff4b4b;color:#1f2937;line-height:1.6;"
                "font-family:monospace;white-space:pre-wrap;'>"
                + final_report.replace("<", "&lt;").replace(">", "&gt;")
                + "</div>",
                unsafe_allow_html=True,
            )

        # 세션에 결과 저장
        st.session_state.ai_report_content  = final_report
        st.session_state.rag_context_used   = rag_context
        st.session_state.graph_context_used = graph_context

    # 5-7b. LangGraph SAR 생성 — 노드별 실시간 진행 상황 표시
    if st.button("🔗 LangGraph SAR 생성 (classify → retrieve → integrate → generate → validate)"):

        _NODE_META = [
            ("classify",  "🏷️ 질의 분류"),
            ("retrieve",  "📚 RAG 검색 (network=Hybrid / standard=Vector)"),
            ("integrate", "🔗 컨텍스트 앙상블"),
            ("generate",  "✍️ Groq LLM SAR 생성"),
            ("validate",  "✅ 법령 근거 검증"),
        ]
        # 조건부 분기 노드는 UI 상 같은 단계로 표시
        _NODE_ALIAS = {"retrieve_vector": "retrieve"}

        st.markdown("**LangGraph 파이프라인**")
        _ph = {name: st.empty() for name, _ in _NODE_META}
        for _n, _lbl in _NODE_META:
            _ph[_n].markdown(f"⬜ {_lbl}")
        _ph["classify"].markdown("⏳ 🏷️ 질의 분류 실행 중...")

        try:
            _inputs = {
                "sar_payload":  sar_payload,
                "sar_json_str": sar_json_str,
                "node_idx":     selected_idx,
            }
            _state: dict = dict(_inputs)
            _completed: list[str] = []
            _node_names = [n for n, _ in _NODE_META]

            for _event in sar_graph.stream(_inputs, stream_mode="updates"):
                _raw_name = next(iter(_event))
                if _raw_name == "__end__":
                    continue

                # 상태 누적
                for _k, _v in _event[_raw_name].items():
                    _state[_k] = _v

                _node_name = _NODE_ALIAS.get(_raw_name, _raw_name)

                # 완료 노드 체크마크
                if _node_name in _ph:
                    _lbl = dict(_NODE_META)[_node_name]
                    _ph[_node_name].markdown(f"✅ {_lbl}")
                    _completed.append(_node_name)

                # 다음 노드 '실행 중' 표시
                if _node_name in _node_names:
                    _next_idx = _node_names.index(_node_name) + 1
                    if _next_idx < len(_NODE_META):
                        _next_name, _next_lbl = _NODE_META[_next_idx]
                        if _next_name not in _completed:
                            _ph[_next_name].markdown(f"⏳ {_next_lbl} 실행 중...")

            # 결과 추출
            final_report = _state.get("final_report", "")
            is_valid     = _state.get("is_valid", False)
            query_type   = _state.get("query_type", "")
            rag_scores   = _state.get("rag_scores") or []

            # 메트릭 표시
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("질의 유형",     query_type)
            _mc2.metric("법령 근거",     "✅ 포함" if is_valid else "⚠️ 미포함")
            if rag_scores:
                _avg_score = sum(r.get("score", 0) for r in rag_scores) / len(rag_scores)
                _mc3.metric("RAG 평균 유사도", f"{_avg_score:.2f}")

            st.markdown(
                "<div style='background:#f0f2f6;padding:20px;border-radius:8px;"
                "border-left:5px solid #6366f1;color:#1f2937;line-height:1.6;"
                "font-family:monospace;white-space:pre-wrap;'>"
                + final_report.replace("<", "&lt;").replace(">", "&gt;")
                + "</div>",
                unsafe_allow_html=True,
            )

            st.session_state.ai_report_content  = final_report
            st.session_state.rag_context_used   = _state.get("rag_context", "")
            st.session_state.graph_context_used = _state.get("graph_context", "")

        except Exception as _exc:
            st.error(f"❌ LangGraph 실행 오류: {_exc}")

    # 5-8. 다운로드 버튼
    if st.session_state.ai_report_content:
        st.markdown("---")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label     = "📄 보고서 다운로드(TXT)",
                data      = st.session_state.ai_report_content,
                file_name = f"SAR_Report_Node_{selected_idx}.txt",
                mime      = "text/plain",
            )
        with dl_col2:
            pdf_result = create_pdf_report(
                report_text = st.session_state.ai_report_content,
                node_id     = selected_idx,
                risk_level  = risk_level,
            )
            if isinstance(pdf_result, str):
                st.error(pdf_result)
            else:
                st.download_button(
                    label     = "📕 PDF 보고서 다운로드",
                    data      = pdf_result,
                    file_name = f"SAR_Report_{selected_idx}.pdf",
                    mime      = "application/pdf",
                )

    # 5-9. 참조 컨텍스트 패널 — 텍스트 RAG(왼쪽) + GraphRAG(오른쪽)
    st.markdown("---")
    ref_col1, ref_col2 = st.columns(2)

    with ref_col1:
        if st.session_state.rag_context_used:
            with st.expander("📚 참조된 KoFIU 법령·지침 (텍스트 RAG)"):
                st.markdown(
                    "<pre style='font-size:0.8rem; white-space:pre-wrap;'>"
                    + st.session_state.rag_context_used
                    + "</pre>",
                    unsafe_allow_html=True,
                )
        elif kr.rag_available:
            with st.expander("📚 텍스트 RAG 컨텍스트"):
                st.info("유사도 임계값(0.3)을 초과하는 관련 법령 구절이 없습니다.")

    with ref_col2:
        if st.session_state.graph_context_used:
            with st.expander("🗄️ 거래 네트워크 분석 결과 (GraphRAG)"):
                st.markdown(
                    "<pre style='font-size:0.8rem; white-space:pre-wrap;'>"
                    + st.session_state.graph_context_used
                    + "</pre>",
                    unsafe_allow_html=True,
                )
        elif kr.graph_available:
            with st.expander("🗄️ GraphRAG 컨텍스트"):
                st.info("Neo4j에서 해당 노드의 네트워크 데이터를 찾을 수 없습니다.")
        else:
            with st.expander("🗄️ GraphRAG (Neo4j 미연결)"):
                st.info(
                    "Neo4j가 연결되지 않아 그래프 네트워크 분석을 건너뜁니다.\n\n"
                    "활성화하려면:\n"
                    "1. Neo4j Desktop에서 인스턴스를 시작하세요.\n"
                    "2. `python tools/neo4j_loader.py` 를 실행하세요."
                )

    # 5-10. 원본 JSON 보기
    with st.expander("📝 원본 분석 JSON 데이터 보기"):
        st.code(sar_json_str, language="json")
