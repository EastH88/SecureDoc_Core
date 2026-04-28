"""SecureDoc Core FastAPI 서버 - PDF 문서 기반 멀티모달 RAG API"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    DocumentInfo,
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# 문서 메타데이터 저장 경로
_META_FILE = Path(settings.UPLOAD_DIR) / "documents.json"


def _load_documents() -> dict[str, dict]:
    if _META_FILE.exists():
        return json.loads(_META_FILE.read_text(encoding="utf-8"))
    return {}


def _save_documents(docs: dict[str, dict]) -> None:
    _META_FILE.parent.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(
        json.dumps(docs, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 모든 서비스 초기화"""
    settings.ensure_dirs()
    logger.info("SecureDoc Core 서버 초기화 시작")

    # 1. 임베더
    from app.ingest.embedder import Embedder
    embedder = Embedder()

    # 2. 저장소
    from app.retrieval.store import BM25Store, VectorStore
    vector_store = VectorStore()
    bm25_store = BM25Store()

    # 3. 검색기
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.sparse import SparseRetriever
    from app.retrieval.hybrid import HybridRetriever
    dense = DenseRetriever(vector_store, embedder)
    sparse = SparseRetriever(bm25_store, embedder)
    hybrid = HybridRetriever(dense, sparse, vector_store)

    # 4. LLM / VLM 클라이언트
    from app.llm.client import LLMClient
    from app.vlm.client import VLMClient
    llm_client = LLMClient()
    vlm_client = VLMClient()

    # 5. Langfuse 트레이서
    from app.observability.langfuse_client import LangfuseTracer
    tracer = LangfuseTracer()

    # 6. 에이전트 노드에 서비스 주입 및 그래프 초기화
    from app.agent.nodes import init_services
    from app.agent.graph import init_graph
    init_services(llm_client, vlm_client, hybrid, tracer)
    init_graph(tracer)

    # 7. 인제스트 파이프라인 컴포넌트
    from app.ingest.parser import PDFParser
    from app.ingest.chunker import SlidingWindowChunker
    parser = PDFParser()
    chunker = SlidingWindowChunker()

    # app.state에 저장
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25_store = bm25_store
    app.state.hybrid = hybrid
    app.state.parser = parser
    app.state.chunker = chunker
    app.state.tracer = tracer

    logger.info("SecureDoc Core 서버 초기화 완료 (port %d)", settings.API_PORT)
    yield
    logger.info("SecureDoc Core 서버 종료")


app = FastAPI(
    title="SecureDoc Core",
    description="PDF 문서 기반 멀티모달 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /upload ─────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """PDF 문서 업로드 → 파싱 → 청킹 → 임베딩 → 저장"""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다.")

    doc_id = str(uuid.uuid4())[:8]
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{doc_id}_{file.filename}"

    # 파일 저장
    content = await file.read()
    file_path.write_bytes(content)
    logger.info("파일 저장: %s (%d bytes)", file_path, len(content))

    try:
        parser: PDFParser = app.state.parser
        chunker: SlidingWindowChunker = app.state.chunker
        embedder: Embedder = app.state.embedder
        vector_store: VectorStore = app.state.vector_store
        bm25_store: BM25Store = app.state.bm25_store

        # 1. 파싱
        page_count = parser.get_page_count(str(file_path))
        text_blocks, image_blocks = parser.parse(str(file_path), doc_id)

        # 2. 청킹
        all_blocks = text_blocks + image_blocks
        chunks = chunker.chunk(all_blocks, doc_id)

        # 3. 임베딩 (figure 청크는 빈 텍스트 → 임베딩은 placeholder)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)

        # 4. 저장
        vector_store.add_chunks(chunks, embeddings)

        # 5. BM25 토큰화 + 인덱싱
        tokenized = [embedder.tokenize_korean(t) for t in texts]
        bm25_store.add_documents(doc_id, chunks, tokenized)

        # 메타데이터 저장
        docs = _load_documents()
        docs[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "uploaded_at": datetime.now().isoformat(),
        }
        _save_documents(docs)

        return UploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            page_count=page_count,
            chunk_count=len(chunks),
            message=f"문서 업로드 및 인덱싱 완료 ({len(chunks)}개 청크)",
        )

    except Exception as e:
        logger.exception("문서 인제스트 실패: %s", file.filename)
        raise HTTPException(500, f"문서 처리 중 오류 발생: {e}")


# ── POST /query ──────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_document(req: QueryRequest):
    """질문 → LangGraph 에이전트 → 답변 + citation"""
    if not req.question.strip():
        raise HTTPException(400, "질문을 입력하세요.")

    from app.agent.graph import run_query
    try:
        response = await run_query(req.question, req.doc_id)
        return response
    except Exception as e:
        logger.exception("질의 처리 실패")
        raise HTTPException(500, f"질의 처리 중 오류 발생: {e}")


# ── GET /documents ───────────────────────────────────────────
@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """업로드된 문서 목록 조회"""
    docs = _load_documents()
    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
            uploaded_at=d["uploaded_at"],
        )
        for d in docs.values()
    ]


# ── 헬스체크 ─────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "SecureDoc Core"}
