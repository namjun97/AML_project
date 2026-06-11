---
title: AML Detection System
emoji: 🕵️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# 🕵️ GNN + XGBoost 하이브리드 자금세탁 탐지 시스템

**그래프 신경망 기반 AML(Anti-Money Laundering) 탐지 · SHAP 설명 · LangGraph + RAG SAR 자동 생성**

PaySim 금융 거래 데이터에서 GraphSAGE와 XGBoost를 결합해 자금세탁 의심 계좌를 찾아내고,
KoFIU 법령 RAG를 기반으로 의심거래보고서(SAR)를 Groq LLM이 실시간 스트리밍으로 생성하는 End-to-End 시스템입니다.

[![CI Pipeline](https://github.com/namjun97/AML_project/actions/workflows/ci.yml/badge.svg)](https://github.com/namjun97/AML_project/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-GraphSAGE-7A1FA2)
![XGBoost](https://img.shields.io/badge/XGBoost-Hybrid-0099CC)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.6.x-4B5563?logo=langchain&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Monitoring-FF6B35?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-llama--3.1-F55036?logo=groq&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphRAG-008CC1?logo=neo4j&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/Vartyor/aml-project)

</div>

---

## 📑 목차

- [왜 이 프로젝트인가](#-왜-이-프로젝트인가)
- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [LangGraph SAR 파이프라인](#-langgraph-sar-파이프라인)
- [LangSmith 모니터링](#-langsmith-모니터링)
- [기술 스택](#️-기술-스택)
- [빠른 시작](#-빠른-시작)
- [프로젝트 구조](#-프로젝트-구조)
- [핵심 트러블슈팅](#-핵심-트러블슈팅)
- [로드맵](#-로드맵)
- [작성자](#-작성자)

---

## 💡 왜 이 프로젝트인가

자금세탁은 **단일 계좌의 이상 거래**가 아니라 **여러 계좌가 얽힌 네트워크 패턴**(레이어링, 분산 송금, 합치기 등)으로 발생합니다. 전통적인 룰 기반 · 단일 ML 모델은 계좌 단위 피처만 보기 때문에 이러한 구조를 놓치기 쉽습니다.

이 프로젝트는 세 가지 문제를 하나의 파이프라인으로 해결합니다.

| 금융 도메인 Pain Point | 해결 접근 |
|---|---|
| **탐지 정확도** — 계좌 네트워크 패턴이 단일 계좌 피처로 포착되지 않음 | **GraphSAGE(GNN) + XGBoost 하이브리드** (74차원 입력) |
| **설명 책임** — 금융 규제상 "왜 의심스러운가"를 근거로 제시해야 함 | **SHAP TreeExplainer** + 원본 피처/GNN 임베딩 기여도 해석 |
| **보고 부담** — 의심거래보고서(SAR) 수작업 생성에 긴 시간 소요 | **Groq LLM + KoFIU 법령 RAG + LangGraph**로 자동 생성 |

---

## 🖥️ 주요 기능

### 1. 위험도 정렬 사이드바 — 의심 계좌 즉시 접근

<p align="center">
  <img src="기능1-사이드바.JPG" alt="위험도 정렬 사이드바" width="820"/>
</p>

XGBoost 사기 확률 기준으로 전체 계좌를 내림차순 정렬합니다. 클릭 한 번으로 분석 대상을 전환할 수 있습니다.

---

### 2. 사기 확률 점수 & 위험 등급 카드

<p align="center">
  <img src="기능2-의심 계좌 점수.JPG" alt="사기 확률 점수" width="820"/>
</p>

노드 ID · 사기 확률(%) · 위험 등급(낮음/중간/높음/매우 높음)을 메트릭 카드로 제시합니다.

---

### 3. SHAP 기여도 분석 — "왜 의심스러운가"

<p align="center">
  <img src="기능3-의심 점수 분석.JPG" alt="SHAP 분석" width="820"/>
</p>

TreeExplainer 기반 **Waterfall 차트**로 각 피처가 사기 판정에 얼마나 기여했는지를 시각화합니다. GNN 임베딩 피처(`GNN_Emb_*`)까지 해석 대상에 포함되어 **네트워크 구조 기여도**까지 수치로 확인할 수 있습니다.

---

### 4. 자금 흐름 네트워크 시각화 — 이중 레이어

<p align="center">
  <img src="기능4-의심 네트워크 시각화.JPG" alt="자금 흐름 네트워크" width="820"/>
</p>

pyvis 기반의 이중 레이어 시각화:

- **실선(파랑)** — 실제 거래 엣지, BFS 2-hop, 화살표 방향과 금액을 선 굵기로 표현
- **점선(보라)** — GNN 임베딩 코사인 KNN 기반 **패턴 유사 계좌**. 직접 거래가 없어도 같은 세탁 링 소속 계좌를 탐지

> PaySim의 평균 degree가 약 0.52로 매우 희소하기 때문에 실거래만으로는 세탁 구조가 드러나지 않는 문제를 이중 레이어로 해결했습니다.

---

### 5. SAR 보고서 AI 실시간 생성 — 스트리밍 & LangGraph

<p align="center">
  <img src="기능5-SAR 보고서 AI 생성.JPG" alt="SAR 생성" width="820"/>
</p>

두 가지 모드로 SAR을 생성합니다.

- **스트리밍 모드** — Groq `llama-3.1-8b-instant` Streaming API로 토큰 단위 실시간 출력
- **LangGraph 모드** — 5개 노드 파이프라인(classify → retrieve → integrate → generate → validate)으로 실행, 법령 근거 포함 여부 자동 검증

KoFIU 자금세탁방지 업무지침 · 가상통화 가이드라인 · 연차보고서를 ChromaDB에서 검색한 컨텍스트를 LLM 프롬프트에 주입하여 **법적 근거가 포함된** SAR을 만듭니다.

- 섹션 I (분석 개요) — Python 고정 양식
- 섹션 II / III / IV — LLM 생성 (`##II## … ##II_END##` 마커 구조)
- 섹션 V — Neo4j GraphRAG 결과 직접 삽입

---

### 6. SAR TXT 다운로드

<p align="center">
  <img src="기능6-다운로드 기능.JPG" alt="TXT 다운로드" width="820"/>
</p>

생성된 SAR을 `SAR-{year}-{node_id:06d}.txt` 형식으로 즉시 다운로드할 수 있습니다. 매 생성마다 동일한 문서 구조가 보장되어(파싱 안정화 3층 메커니즘) 외부 워크플로에 그대로 연결 가능합니다.

---

### 7. 모델 · RAG 산출물 관리

<p align="center">
  <img src="기능7-산출물.JPG" alt="산출물 관리" width="820"/>
</p>

학습된 GNN(`gnn_model.pth`) · XGBoost(`fraud_model.pkl`) · ChromaDB 벡터스토어(`chroma_db/`)가 모두 프로젝트 내부에 포함되어 있어 **추가 학습 없이 즉시 추론**이 가능합니다.

---

## 🧭 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  ① DATA LAYER — PaySim 전처리 & 그래프 구성                     │
│     TRANSFER/CASH_OUT 필터 → 계좌 노드(10D) + 거래 엣지(6D)     │
│     earliest_step 기반 시계열 분할 (70/10/20)                   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  ② GRAPH LEARNING — GraphSAGE 3층 (PyTorch Geometric)          │
│     SAGEConv × 3 + Dropout(0.3)                               │
│     10D → 128 → 128 → 64D 노드 임베딩                           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  ③ CLASSIFICATION + EXPLANATION — XGBoost + SHAP                │
│     입력: GNN 임베딩 64D + 원본 피처 10D = 74D                  │
│     scale_pos_weight 동적 계산 · TreeExplainer 폴백 체인         │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  ④ KNOWLEDGE LAYER — Text RAG + GraphRAG                        │
│     LangChain + ChromaDB (PersistentClient)                    │
│       └ KoFIU 지침 3종 → sentence-transformers 임베딩           │
│     Neo4j Cypher Q1~Q4 → 자금 흐름 경로 · 허브 계좌 지표          │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  ⑤ SAR PIPELINE — LangGraph 5-Node Workflow                     │
│     [classify] → [retrieve] → [integrate]                      │
│                → [generate] → [validate]                       │
│     LangSmith로 노드별 실행 시간 · 토큰 사용량 추적               │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  ⑥ SERVING LAYER — Streamlit Dashboard                          │
│     위험 노드 정렬 → SHAP Waterfall → pyvis 이중 네트워크         │
│     → SAR 스트리밍 생성 또는 LangGraph 파이프라인 → TXT 다운로드  │
└──────────────────────────────────────────────────────────────┘
```

### 설계 원칙

- **SRP + 의존성 주입(DI)** — `ContextBuilder`, `ReportRunner` 등 각 클래스는 단일 책임을 가지며 외부에서 주입된 의존 객체를 사용
- **Graceful Degradation** — Neo4j 연결 실패 또는 RAG 초기화 실패 시에도 텍스트 RAG · GNN 파이프라인은 정상 동작
- **캐싱** — `@st.cache_resource`로 모델 · KB · LangGraph 그래프는 앱 생명주기 동안 1회 초기화
- **관찰 가능성** — LangSmith로 LangChain 체인 및 LangGraph 노드별 실행 추적

---

## 🔗 LangGraph SAR 파이프라인

### LangChain → LangGraph 전환 이유

기존 LangChain 단순 체인 방식에서는 두 가지 문제가 있었습니다.

1. **단계 격리 불가** — `ContextBuilder → stream → finalize`가 하나의 연속 흐름으로 엮여 있어 중간 단계를 개별 테스트하거나 LangSmith에서 노드 단위로 추적하기 어려웠습니다.
2. **검증 로직 부재** — 생성된 SAR에 법령 근거가 실제로 포함됐는지 확인하는 단계가 없었습니다.

LangGraph의 `State + Node` 구조로 전환하면서 각 단계를 독립적으로 테스트·모니터링할 수 있게 됐고, `validate` 노드에서 법령 키워드 포함 여부를 자동으로 검증합니다.

### 노드 구조

```
[START]
   ↓
[classify]   — sar_payload의 GNN 피처 유무로 "network" / "standard" 분류
   ↓
[retrieve]   — Vector RAG(ChromaDB) + Graph RAG(Neo4j) 병렬 호출
   ↓
[integrate]  — 두 컨텍스트를 앙상블하여 integrated_context 생성
   ↓
[generate]   — Groq LLM으로 SAR 섹션 II~IV 스트리밍 생성
   ↓
[validate]   — 법령 키워드(특정금융정보법, KoFIU 등) 포함 여부 검증 + 최종 조립
   ↓
[END]
```

### State 스키마

```python
class AMLState(TypedDict):
    sar_payload:         dict   # build_sar_payload() 결과
    sar_json_str:        str    # JSON 직렬화 문자열
    node_idx:            int    # 분석 대상 노드 인덱스
    query_type:          str    # "network" | "standard"
    rag_context:         str    # Vector RAG 검색 결과
    graph_context:       str    # Graph RAG 검색 결과
    integrated_context:  str    # 두 컨텍스트 앙상블 텍스트
    sar_draft:           str    # LLM 생성 SAR 초안
    final_report:        str    # 검증·조립 완성 보고서
    is_valid:            bool   # 법령 근거 포함 여부
```

### 사용 예시

```python
from reporters.sar_graph import build_sar_graph

graph = build_sar_graph(context_builder, report_runner)
result = graph.invoke({
    "sar_payload":  sar_payload,
    "sar_json_str": sar_json_str,
    "node_idx":     node_idx,
})
print(result["final_report"])
print("법령 근거 포함:", result["is_valid"])   # True / False
print("질의 유형:", result["query_type"])       # "network" / "standard"
```

---

## 📊 LangSmith 모니터링

LangSmith를 통해 LangChain RAG 체인과 LangGraph 노드 실행을 추적합니다.

### 설정

`.env` 파일에 아래 값을 추가하면 앱 시작 시 자동으로 추적이 활성화됩니다.

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...          # smith.langchain.com 에서 발급
LANGCHAIN_PROJECT=AML-project
```

### 추적 대상

| 추적 항목 | 내용 |
|---|---|
| **RAG 체인** | ChromaDB 검색 쿼리 · 검색 결과 · 응답 시간 |
| **LangGraph 노드** | classify / retrieve / integrate / generate / validate 각 단계 입출력 |
| **LLM 호출** | Groq llama-3.1 토큰 사용량 · 지연 시간 |

> LangSmith API 키는 [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys 에서 발급합니다.

---

## 🛠️ 기술 스택

**Graph & ML**  `PyTorch` · `PyTorch Geometric` · `GraphSAGE(SAGEConv)` · `XGBoost` · `SHAP` · `scikit-learn`
**LLM & RAG**  `LangChain 0.3.x` · `LangGraph 0.6.x` · `LangSmith` · `ChromaDB` · `Groq(llama-3.1)` · `sentence-transformers` · `Neo4j`
**Data & Viz**  `pandas` · `numpy` · `pyvis` · `matplotlib` · `PyPDF2`
**Serving**  `Streamlit` · `Python 3.11` · `Git / GitHub`

---

## 🚀 빠른 시작

### 사전 요구사항

- Python 3.11+
- [Groq API 키](https://console.groq.com) (무료, SAR 생성에 사용)
- (선택) Neo4j 5.x — GraphRAG 기능 사용 시
- (선택) [LangSmith API 키](https://smith.langchain.com) — 파이프라인 모니터링 시

### 설치 & 실행 (5분)

```bash
# 1) 저장소 클론
git clone https://github.com/namjun97/AML_project.git
cd AML_project

# 2) 가상환경
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3) 의존성 설치
pip install -r requirements.txt

# 4) 환경변수 설정
cp .env.example .env
# .env 파일을 열어 GROQ_API_KEY 등 실제 값 입력

# 5) 대시보드 실행
streamlit run app.py
# → http://localhost:8501
```

### (선택) GraphRAG 활성화 — Neo4j AuraDB 연결

```bash
# .env 파일에 Neo4j 연결 정보 입력
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>

# 스키마 생성 & 데이터 적재
python tools/neo4j_schema.py
python tools/neo4j_loader.py
# → app.py 재실행 시 GraphRAG 자동 활성화 (미연결 시 Graceful Degradation)
```

### 재학습이 필요한 경우

```bash
jupyter notebook aml_project.ipynb            # 1) 전처리 & 그래프 구축
jupyter notebook aml_model_comparison.ipynb   # 2) GraphSAGE + XGBoost 학습
# → gnn_model.pth, fraud_model.pkl 갱신
```

---

## 📂 프로젝트 구조

```
AML_project/
├── app.py                          # Streamlit UI (순수 프레젠테이션 레이어)
├── requirements.txt
├── .env.example                    # 환경변수 템플릿 (커밋용)
│
├── models/
│   └── embedding_extractor.py      # GraphSAGE 3층 (EmbeddingExtractor)
│
├── analysis/
│   ├── shap_analyzer.py            # TreeExplainer + KernelExplainer 폴백
│   └── network_visualizer.py       # 실거래 + GNN KNN 이중 레이어 시각화
│
├── knowledge/
│   ├── rag_knowledge_base.py       # LangChain + ChromaDB(PersistentClient)
│   └── graph_rag.py                # Neo4j Cypher Q1~Q4 (GraphRAG)
│
├── reporters/
│   ├── ai_report_generator.py      # Groq 스트리밍 SAR 생성
│   ├── sar_template.py             # 3층 안정화: 이스케이프·정규화·고정 양식
│   ├── context_builder.py          # Text RAG + GraphRAG 컨텍스트 조립 (DI)
│   ├── report_runner.py            # 스트리밍 오케스트레이터
│   ├── sar_graph.py                # LangGraph 5-Node SAR 파이프라인
│   └── pdf_report_generator.py     # (선택) PDF 보고서 변환
│
├── loaders/
│   ├── resource_loader.py          # @st.cache_resource 모델 캐싱
│   └── knowledge_loader.py         # Graceful Degradation
│
├── tools/
│   ├── extract_pdf_to_txt.py       # 벡터 아웃라인 PDF → TXT 변환
│   ├── neo4j_schema.py             # Neo4j 인덱스 & 제약 생성
│   └── neo4j_loader.py             # PaySim → Neo4j 적재
│
├── knowledge_base/                 # KoFIU 지침 원본 (PDF/TXT)
├── chroma_db/                      # ChromaDB 벡터스토어 (persistent)
├── dataset/                        # PaySim CSV 원본
│
├── aml_project.ipynb               # EDA · 전처리 · 그래프 구축
├── aml_model_comparison.ipynb      # GraphSAGE 학습 + XGBoost 하이브리드
│
├── gnn_model.pth                   # 학습된 GNN 가중치
├── fraud_model.pkl                 # 학습된 XGBoost 모델
│
└── portfolio.html                  # 취업용 포트폴리오 페이지
```

---

## 🔧 핵심 트러블슈팅

직접 디버깅하고 코드로 해결한 9가지 이슈입니다. 상세한 *문제 → 원인 → 해결 → 코드* 흐름은 [`portfolio.html`](portfolio.html)의 **03 Troubleshooting** 섹션에서 확인할 수 있습니다.

| # | 문제 | 원인 | 해결책 |
|---|---|---|---|
| **TS-01** | SAR 출력 구조가 매번 달라짐 | LLM 비결정성 + 마커 마크다운 변형 + `str.format()` 충돌 | `temperature=0.1, seed=42` + `_normalize_markers()` + `_safe_section()` 3층 안정화 |
| **TS-02** | ChromaDB 초기화 오류 | LangChain 내부 `chromadb.Client()` deprecated | `PersistentClient`를 직접 생성 후 `client=` 인자로 주입 |
| **TS-03** | SHAP `TreeExplainer` 예외 | XGBoost `base_score`가 `"[0.5]"` 문자열 배열로 저장 | `save_config()` JSON 직접 수정 → `load_config()` + KernelExplainer 폴백 |
| **TS-04** | 그래프 시각화 무의미 (degree≈0.52) | PaySim 희소성으로 세탁 링 구조 미노출 | GNN 임베딩 코사인 KNN **이중 레이어** 시각화 |
| **TS-05** | Groq LLM 타임아웃 | 대형 LLM 전체 응답 시간이 기본 timeout 초과 | `stream=True` + Groq SDK 스트리밍 모드로 전환 |
| **TS-06** | 비현실적으로 높은 Val AUC | 시간 역전 + 라벨 누수 + 스케일러 누수 + 다중공선성 | 4단계 개별 방어선 구축(시계열 분할, 파생 피처 배제, train-only fit, `hour_of_day_norm` 직접 계산) |
| **TS-07** | Cypher `avg(count())` 오류 | Aggregation nested inside aggregation | 3-Step Cypher 분리 (전체 평균 → 대상 계좌 → 배율) |
| **TS-08** | `pip install` 무한 백트래킹 | `langchain-community 0.3.20`이 `langchain-core>=0.3.45`를 요구하는데 `0.3.29`로 핀하여 충돌, 추가로 `langsmith<0.4` 상한과 `langchain-core 0.3.86`의 `langsmith>=0.3.45` 요구가 교차 | 각 패키지 의존성 제약의 교집합을 수동으로 계산해 `langchain-core==0.3.86`, `langsmith==0.3.45`로 일관 핀 |
| **TS-09** | `requirements.txt` 한글 주석 UnicodeDecodeError | Windows pip이 cp949로 파일을 읽다가 UTF-8 멀티바이트 문자(한글, `──`) 를 디코딩 실패 | 주석을 전부 ASCII 영문으로 교체 |

---

## 🧪 모델 학습 개요

### 데이터 분할 (누수 차단)

```python
# 계좌별 첫 등장 시점으로 시계열 정렬
account_first_step = pd.concat([
    df[['nameOrig', 'step']].rename(columns={'nameOrig': 'account'}),
    df[['nameDest', 'step']].rename(columns={'nameDest': 'account'})
]).groupby('account')['step'].min()

account_df['earliest_step'] = account_df['account'].map(account_first_step)
sorted_order = account_df['earliest_step'].argsort(kind='stable').values
# 초기 70% = train / 중간 10% = val / 후기 20% = test
```

### 하이브리드 입력 구성

```python
# GNN 가중치 freeze 후 임베딩만 추출
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)   # [N, 64]

# 원본 피처와 결합 → XGBoost 입력
X_combined = np.hstack([embeddings.numpy(), node_features_scaled])  # [N, 74]
```

---

## 🗺️ 로드맵

- [x] **Phase 1** — PaySim → PyG 그래프 변환 → GraphSAGE 학습
- [x] **Phase 2** — GNN 임베딩 + 원본 피처 → XGBoost 하이브리드 + SHAP 분석
- [x] **Phase 3** — LangChain + ChromaDB RAG + Groq 스트리밍 SAR 생성
- [x] **Phase 4** — Neo4j GraphRAG (Q1~Q4) + SAR 템플릿 안정화 리팩토링
- [x] **Phase 5** — LangGraph 5-Node SAR 파이프라인 + LangSmith 모니터링 연동
- [ ] **Phase 6** — GraphRAG 고도화: GNN 임베딩 기반 유사 계좌 클러스터를 RAG 컨텍스트로 활용
- [ ] **Phase 7** — 실시간 스트리밍 입력 (Kafka 등) + 증분 학습 고려

---

## 🙋 작성자

**김남준**
데이터 분석 · AI 모델링에 관심을 둔 신입 개발자. 문제 정의부터 배포까지 전 과정을 1인이 책임지는 End-to-End 개발을 지향합니다.

- 🤗 [라이브 데모 (HuggingFace Spaces)](https://huggingface.co/spaces/Vartyor/aml-project)
- 📧 varute1997@gmail.com

---

## 📝 라이선스

본 저장소의 코드는 포트폴리오 목적으로 공개됩니다.
PaySim 데이터셋 · KoFIU 지침 원본 문서는 각 원저작자의 라이선스를 따릅니다.

<div align="center">
<sub>Built with <b>Python</b> · <b>PyTorch Geometric</b> · <b>XGBoost</b> · <b>LangChain</b> · <b>LangGraph</b> · <b>LangSmith</b> · <b>Groq</b> · <b>Neo4j</b> · <b>Streamlit</b></sub>
</div>
