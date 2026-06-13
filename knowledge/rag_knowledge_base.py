from __future__ import annotations

import warnings
from pathlib import Path

import chromadb
from PyPDF2 import PdfReader

# ── LangChain imports ──────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import Chroma as LangChainChroma

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_embeddings_backend(
    embed_model: str = "nomic-embed-text",
    ollama_base_url: str = "http://localhost:11434",
):
    """
    임베딩 백엔드를 자동으로 선택합니다.

    우선순위:
    1. HuggingFaceEmbeddings (sentence-transformers) — API 키 불필요, 로컬 실행
    2. OllamaEmbeddings — Ollama 서버가 실행 중인 경우 사용 (로컬 개발 환경)

    embed_model이 'nomic-embed-text'처럼 Ollama 전용 모델명일 경우
    HuggingFace 모드에서는 다국어 지원 모델로 자동 대체합니다.
    """
    # ── 1순위: HuggingFaceEmbeddings (sentence-transformers) ─────────────
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        # Ollama 전용 모델명이면 다국어 지원 sentence-transformer로 대체
        # MiniLM-L12 (~120MB) 선택 이유:
        #   - mpnet-base-v2 (~420MB) 대비 메모리 3.5배 절약
        #   - 한국어 포함 50개 언어 지원, AML 문서 검색 품질 충분
        #   - 16GB RAM 환경에서 Neo4j + PyTorch 동시 실행 시 OOM 방지
        hf_model = embed_model
        if embed_model in ("nomic-embed-text", "mxbai-embed-large", "all-minilm"):
            hf_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        return HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        pass

    # ── 2순위: OllamaEmbeddings (기존 로컬 개발 환경 호환) ─────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
    return OllamaEmbeddings(
        model=embed_model,
        base_url=ollama_base_url,
    )


# 상수

_CHUNK_SIZE    = 500   # 청크당 최대 문자 수
_CHUNK_OVERLAP = 100   # 청크 간 겹침 문자 수
_COLLECTION    = "aml_knowledge"


# KnowledgeBase 클래스
class KnowledgeBase: # Langchain 기반 RAG 지식베이스 클래스
    def __init__(
        self,
        pdf_directory: str = "knowledge_base",
        persist_dir: str = "chroma_db",
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        self.pdf_directory   = Path(pdf_directory)
        self.persist_dir     = persist_dir
        self.embed_model     = embed_model
        self.ollama_base_url = ollama_base_url

        # LangChain 컴포넌트 (지연 초기화)
        self._embeddings = None
        self._vectorstore: LangChainChroma | None = None

        # RecursiveCharacterTextSplitter — 한국어 문서에 적합한 구분자 설정
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )


    # 내부 초기화 (지연 로딩)

    def _get_embeddings(self):
        """임베딩 인스턴스를 반환합니다 (최초 1회 생성).

        HuggingFaceEmbeddings(sentence-transformers) 우선 사용.
        미설치 시 OllamaEmbeddings로 자동 폴백합니다.
        """
        if self._embeddings is None:
            self._embeddings = _get_embeddings_backend(
                embed_model=self.embed_model,
                ollama_base_url=self.ollama_base_url,
            )
        return self._embeddings

    def _get_vectorstore(self) -> LangChainChroma: # LangChain Chroma 벡터스토어를 반환(최초 1회 생성)
        if self._vectorstore is None:
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            embeddings = self._get_embeddings()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    self._vectorstore = LangChainChroma(
                        collection_name=_COLLECTION,
                        embedding_function=embeddings,
                        client=chroma_client,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                except Exception as e:
                    # 임베딩 모델 교체로 차원 불일치 발생 시 컬렉션을 초기화하고 재생성
                    if "dimension" in str(e).lower() or "InvalidDimensionException" in str(type(e)):
                        print(
                            "[KnowledgeBase] [WARN] 임베딩 차원 불일치 감지 — "
                            "기존 컬렉션을 삭제하고 재빌드합니다."
                        )
                        try:
                            chroma_client.delete_collection(_COLLECTION)
                        except Exception:
                            pass
                        self._vectorstore = LangChainChroma(
                            collection_name=_COLLECTION,
                            embedding_function=embeddings,
                            client=chroma_client,
                            collection_metadata={"hnsw:space": "cosine"},
                        )
                    else:
                        raise
        return self._vectorstore


    # 텍스트 추출 (정적 메서드)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, str]: # PDF에서 텍스트 추출
        path = Path(pdf_path)
        try:
            reader       = PdfReader(str(path))
            pages        = []
            failed_pages = 0
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    failed_pages += 1
            full_text = "\n".join(pages).strip()
            status = (
                f"OK ({len(reader.pages)}p, 실패 {failed_pages}p)"
                if full_text
                else f"빈 텍스트 ({len(reader.pages)}p — 벡터 아웃라인 또는 스캔 PDF)"
            )
            return full_text, status
        except Exception as e:
            return "", f"예외: {e}"

    @staticmethod
    def extract_text_from_txt(txt_path: str | Path) -> str: # TXT 파일에서 문자 반환
        path = Path(txt_path)
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                return path.read_text(encoding=enc).strip()
            except (UnicodeDecodeError, LookupError):
                continue
        return ""


    # 공개 메서드

    def is_built(self) -> bool: # ChromaDB 컬렉션에 이미 벡터화된 문서가 있는지 확인
        try:
            return self._get_vectorstore()._collection.count() > 0
        except Exception:
            return False

    def build(self, force: bool = False) -> None: # 내 문서를 청크 단위로 벡터화해 ChromaDB에 저장
        if not force and self.is_built():
            print("[KnowledgeBase] 기존 지식베이스 감지 → 재빌드 건너뜀.")
            return

        pdf_files = {p.stem: p for p in sorted(self.pdf_directory.glob("*.pdf"))}
        txt_files = {p.stem: p for p in sorted(self.pdf_directory.glob("*.txt"))}
        all_stems = sorted(set(pdf_files) | set(txt_files))

        if not all_stems:
            print(f"[KnowledgeBase] '{self.pdf_directory}' 에 처리할 문서가 없습니다.")
            return

        vectorstore  = self._get_vectorstore()
        total_chunks = 0
        ok_count     = 0
        skip_count   = 0

        for stem in all_stems:
            text   = ""
            source = ""

            # 1) TXT 우선
            if stem in txt_files:
                text   = self.extract_text_from_txt(txt_files[stem])
                source = txt_files[stem].name
                if text:
                    print(f"  [TXT] 처리: {source}")

            # 2) PDF 폴백
            if not text and stem in pdf_files:
                pdf_text, status = self.extract_text_from_pdf(pdf_files[stem])
                source = pdf_files[stem].name
                if pdf_text:
                    text = pdf_text
                    print(f"  [PDF] 추출: {source} [{status}]")
                else:
                    print(
                        f"  [WARN] PDF 텍스트 추출 불가: {source} [{status}]\n"
                        f"     -> tools/extract_pdf_to_txt.py 실행 후 '{stem}.txt' 생성"
                    )

            if not text:
                skip_count += 1
                continue

            # ── LangChain RecursiveCharacterTextSplitter 청킹 ─────────
            lc_docs: list[Document] = self._splitter.create_documents(
                texts=[text],
                metadatas=[{"source": source}],
            )

            # ── 청크 ID 생성 (upsert를 위해 결정적 ID 사용) ───────────
            ids = [f"{stem}__chunk_{i}" for i in range(len(lc_docs))]

            # ── LangChain Chroma add_documents ────────────────────────
            vectorstore.add_documents(documents=lc_docs, ids=ids)
            total_chunks += len(lc_docs)
            ok_count     += 1
            print(f"  [OK] 저장 완료: {source}  ({len(lc_docs)} 청크)")

        print(
            f"[KnowledgeBase] 완료 — 처리 {ok_count}개 / "
            f"건너뜀 {skip_count}개 / 총 {total_chunks}개 청크"
        )

    def search(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict]:
        results: list[tuple[Document, float]] = (
            self._get_vectorstore().similarity_search_with_score(query, k=k)
        )
        return [
            {
                "text":   doc.page_content,
                "source": doc.metadata.get("source", "알 수 없음"),
                # LangChain Chroma는 cosine distance 반환 → 1-dist = similarity
                "score":  round(1.0 - float(score), 4),
            }
            for doc, score in results
        ]

    def format_context(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.3,
    ) -> str:
        results  = self.search(query, k=k)
        filtered = [r for r in results if r["score"] >= score_threshold]

        if not filtered:
            return ""

        parts = []
        for i, r in enumerate(filtered, 1):
            parts.append(
                f"[참고 {i}] 출처: {r['source']} (유사도: {r['score']:.2f})\n"
                f"{r['text']}"
            )
        return "\n\n".join(parts)
