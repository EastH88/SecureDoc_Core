"""SecureDoc Core 데이터 모델 정의"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field


# ── 문서 메타데이터 ──────────────────────────────────────────
class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int = 0
    uploaded_at: datetime = Field(default_factory=datetime.now)


# ── 청크 ────────────────────────────────────────────────────
class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    page_num: int
    bbox: list[float] | None = None
    chunk_type: str = "text"          # text / table / figure
    image_path: str | None = None


class Chunk(BaseModel):
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = None


# ── 질의 / 응답 ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    doc_id: str | None = None


class QueryType(str, enum.Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    VISUAL = "visual"


class Citation(BaseModel):
    chunk_id: str
    page_num: int
    content_preview: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    faithfulness_score: float = 0.0
    query_type: str = "simple"
    trace_id: str | None = None


# ── 업로드 응답 ──────────────────────────────────────────────
class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str


# ── 문서 목록 ────────────────────────────────────────────────
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    uploaded_at: str


# ── LangGraph 에이전트 상태 ──────────────────────────────────
class AgentState(TypedDict, total=False):
    question: str
    doc_id: str | None
    query_type: str
    chunks: list[dict]
    visual_context: str
    answer: str
    citations: list[dict]
    faithfulness_score: float
    retry_count: int
    trace_id: str | None
