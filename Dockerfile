# ──────────────────────────────────────────────────────
# HF Spaces Docker SDK
# - 포트: 7860 (HF Spaces 필수)
# - 유저: non-root (uid=1000)
# ──────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ── 1단계: CPU-only PyTorch (별도 인덱스 URL 필요) ────────────────────
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir torch-geometric==2.5.2

# ── 2단계: 기본 ML/데이터 패키지 ──────────────────────────────────────
RUN pip install --no-cache-dir \
    streamlit==1.41.0 \
    pandas \
    numpy \
    xgboost \
    joblib \
    scikit-learn \
    shap \
    matplotlib \
    requests \
    PyPDF2 \
    fpdf2 \
    pytz \
    python-dateutil \
    "pyvis==0.3.2"

# ── 3단계: LLM / RAG ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "groq>=0.9.0" \
    chromadb

RUN pip install --no-cache-dir \
    "sentence-transformers>=2.6.0" \
    "langchain-huggingface>=0.0.3" \
    "langchain-text-splitters>=0.2.0"

# ── 4단계: GraphRAG (Neo4j + LangChain) ───────────────────────────────
RUN pip install --no-cache-dir \
    "neo4j>=5.0.0" \
    "langchain-neo4j>=0.3.0" \
    "langchain-community>=0.3.1" \
    "langchain-core>=0.3.0"

# ── RAG 지식베이스 사전 빌드 (콜드 스타트 단축) ───────────────────────
# 1) HF_HOME 을 이미지 내부로 고정 → 임베딩 모델(MiniLM-L12, 다국어)을
#    빌드 타임에 다운로드해 캐시로 굽는다 (런타임 다운로드 제거)
# 2) PDF 파싱 + 청킹 + CPU 임베딩 + ChromaDB 빌드를 빌드 타임에 수행
#    → 런타임에는 kb.is_built() == True 로 재빌드를 건너뛴다
# 주의: 이 레이어는 knowledge/ 또는 knowledge_base/ 변경 시에만 재빌드되도록
#       app.py 등 나머지 소스 복사보다 앞에 배치한다 (레이어 캐시 활용)
ENV HF_HOME=/app/.cache/huggingface

COPY knowledge/ ./knowledge/
COPY knowledge_base/ ./knowledge_base/

RUN python -c "\
from knowledge.rag_knowledge_base import KnowledgeBase; \
kb = KnowledgeBase(pdf_directory='knowledge_base', persist_dir='chroma_db'); \
kb.build(); \
assert kb.is_built(), 'ChromaDB prebuild failed'; \
print('[Docker] ChromaDB prebuild OK')"

# ── 앱 소스 복사 ──────────────────────────────────────────────────────
COPY app.py .
COPY analysis/ ./analysis/
COPY loaders/ ./loaders/
COPY models/ ./models/
COPY reporters/ ./reporters/
COPY tools/ ./tools/
COPY neo4j_config.py .
COPY NanumGothic*.ttf ./
COPY fraud_model.pkl .
COPY gnn_model.pth .
COPY model/ ./model/

# Streamlit 설정
RUN mkdir -p /app/.streamlit
COPY .streamlit/ ./.streamlit/

# HF Spaces non-root 유저
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
