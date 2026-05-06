"""하이브리드 검색 - Dense + Sparse + RRF Fusion + Cross-Encoder Reranker"""

import logging
from collections import defaultdict

from app.config import settings
from app.models import Chunk, ChunkMetadata
from app.retrieval.dense import DenseRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import SparseRetriever
from app.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Dense + Sparse 검색 결과를 RRF로 결합한 뒤 cross-encoder로 재정렬.

    파이프라인:
      ① Dense + Sparse 각각 fetch_k개 후보 수집
      ② RRF 스코어로 융합 → 상위 TOP_K_RETRIEVE개 후보 추림
      ③ Reranker(cross-encoder)로 (query, chunk) 쌍을 직접 점수화해 재정렬
      ④ 최종 top_k개 반환 (LLM 컨텍스트로 사용)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._store = vector_store
        self._reranker = reranker
        self._rrf_k = settings.RRF_K

    def search(
        self,
        query: str,
        doc_id: str | None = None,
        top_k: int | None = None,
    ) -> list[Chunk]:
        """하이브리드 검색 → RRF 융합 → Reranker 재정렬 → 상위 top_k"""
        if not query.strip():
            return []

        top_k = top_k or settings.TOP_K
        retrieve_k = max(settings.TOP_K_RETRIEVE, top_k)
        fetch_k = retrieve_k * 2  # dense/sparse 각각 더 넓게 긁어옴

        # 1. Dense + Sparse 검색
        dense_results = self._dense.search(query, doc_id=doc_id, top_k=fetch_k)
        sparse_results = self._sparse.search(query, doc_id=doc_id, top_k=fetch_k)

        # 2. RRF 스코어 계산
        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, (chunk_id, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank)
        for rank, (chunk_id, _) in enumerate(sparse_results, start=1):
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank)

        # 3. RRF 상위 retrieve_k 후보 추림
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [cid for cid, _ in sorted_ids[:retrieve_k]]

        # 4. ChromaDB에서 청크 내용 조회 (RRF 순서 유지)
        candidates = self._fetch_chunks(candidate_ids)

        # 5. Reranker가 활성화되면 재정렬 후 top_k, 아니면 RRF 상위 top_k 그대로
        if self._reranker is not None and self._reranker.enabled:
            final = self._reranker.rerank(query, candidates, top_k=top_k)
            logger.info(
                "하이브리드 검색 완료: Dense=%d, Sparse=%d → RRF top-%d → Rerank top-%d",
                len(dense_results), len(sparse_results), len(candidates), len(final),
            )
        else:
            final = candidates[:top_k]
            logger.info(
                "하이브리드 검색 완료: Dense=%d, Sparse=%d → RRF top-%d (rerank off)",
                len(dense_results), len(sparse_results), len(final),
            )
        return final

    def _fetch_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """ChromaDB에서 chunk_id 순서대로 청크 내용 조회"""
        chunks: list[Chunk] = []
        for chunk_id in chunk_ids:
            result = self._store.collection.get(
                ids=[chunk_id], include=["documents", "metadatas"]
            )
            if not result["ids"]:
                continue
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
        return chunks
