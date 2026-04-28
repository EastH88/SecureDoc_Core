"""하이브리드 검색 - Dense + Sparse + RRF Fusion"""

import logging
from collections import defaultdict

from app.config import settings
from app.models import Chunk, ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + Sparse 검색 결과를 RRF(Reciprocal Rank Fusion)로 결합

    RRF 스코어 = sum(1 / (k + rank)) where k = 60
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        vector_store: VectorStore,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._store = vector_store
        self._rrf_k = settings.RRF_K

    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """하이브리드 검색 → RRF 융합 → 상위 top_k 청크 반환"""
        if not query.strip():
            return []

        top_k = top_k or settings.TOP_K
        fetch_k = top_k * 3

        # 1. Dense + Sparse 검색
        dense_results = self._dense.search(query, doc_id=doc_id, top_k=fetch_k)
        sparse_results = self._sparse.search(query, doc_id=doc_id, top_k=fetch_k)

        # 2. RRF 스코어 계산
        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank)
        for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank)

        # 3. 스코어 기준 정렬 → 상위 top_k
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = sorted_ids[:top_k]

        # 4. ChromaDB에서 청크 내용 조회
        chunks: list[Chunk] = []
        for chunk_id, rrf_score in top_ids:
            result = self._store.collection.get(
                ids=[chunk_id], include=["documents", "metadatas"]
            )
            if result["ids"]:
                meta = result["metadatas"][0]
                chunks.append(Chunk(
                    content=result["documents"][0],
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=meta.get("doc_id", ""),
                        page_num=meta.get("page_num", 0),
                        chunk_type=meta.get("chunk_type", "text"),
                        image_path=meta.get("image_path") or None,
                    ),
                ))

        logger.info(
            "하이브리드 검색 완료: Dense=%d, Sparse=%d → RRF top-%d",
            len(dense_results), len(sparse_results), len(chunks),
        )
        return chunks
