"""Sparse 검색 - Kiwi 형태소 분석 + BM25 역색인 검색"""

import logging

from app.ingest.embedder import Embedder
from app.retrieval.store import BM25Store

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Kiwi 형태소 분석 기반 BM25 스파스 검색"""

    def __init__(self, bm25_store: BM25Store, embedder: Embedder) -> None:
        self._store = bm25_store
        self._embedder = embedder

    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """쿼리 → 형태소 토큰화 → BM25 검색 → (chunk_id, score) 리스트"""
        if not query.strip():
            return []
        tokens = self._embedder.tokenize_korean(query)
        if not tokens:
            logger.warning("토큰화 결과 비어 있음: '%s'", query)
            return []
        results = self._store.search(tokens, doc_id=doc_id, top_k=top_k)
        logger.debug("Sparse 검색 완료: %d개 결과", len(results))
        return results
