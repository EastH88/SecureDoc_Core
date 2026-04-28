"""Dense 검색 - ChromaDB 벡터 유사도 기반 검색"""

import logging

from app.ingest.embedder import Embedder
from app.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """BGE-M3 Dense 임베딩 기반 벡터 검색"""

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._store = vector_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """쿼리 → Dense 검색 → (chunk_id, score) 리스트

        ChromaDB distance를 유사도 점수로 변환 (1 - distance).
        """
        if not query.strip():
            return []
        query_embedding = self._embedder.embed_query(query)
        results = self._store.search(query_embedding, doc_id=doc_id, top_k=top_k)

        scored: list[tuple[str, float]] = []
        for chunk_id, distance, _content, _meta in results:
            score = max(0.0, 1.0 - distance)
            scored.append((chunk_id, score))

        logger.debug("Dense 검색 완료: %d개 결과", len(scored))
        return scored
