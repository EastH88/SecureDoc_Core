"""Cross-encoder Reranker - 검색 후보를 의미적 관련도 기준으로 재정렬"""

import logging

from sentence_transformers import CrossEncoder

from app.config import settings
from app.models import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """BGE Reranker 기반 cross-encoder. 쿼리-문서 쌍을 직접 점수화해 재정렬"""

    def __init__(self) -> None:
        self._enabled = settings.RERANK_ENABLED
        self._model: CrossEncoder | None = None
        if not self._enabled:
            logger.info("Reranker 비활성화 (RERANK_ENABLED=false)")
            return
        try:
            logger.info("Reranker 모델 로딩: %s", settings.RERANK_MODEL)
            self._model = CrossEncoder(settings.RERANK_MODEL, max_length=512)
            logger.info("Reranker 모델 로딩 완료")
        except Exception:
            logger.exception("Reranker 모델 로딩 실패 — 재정렬 단계를 건너뜁니다")
            self._model = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[Chunk]:
        """쿼리-청크 쌍을 cross-encoder로 점수화해 내림차순 정렬"""
        if not chunks:
            return []
        if not self.enabled:
            return chunks if top_k is None else chunks[:top_k]

        pairs = [(query, c.content) for c in chunks]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            logger.exception("Reranker 추론 실패 — 원본 순서 유지")
            return chunks if top_k is None else chunks[:top_k]

        scored = sorted(
            zip(chunks, scores), key=lambda x: float(x[1]), reverse=True
        )
        ordered = [c for c, _ in scored]
        if top_k is not None:
            ordered = ordered[:top_k]
        logger.info(
            "Reranker 재정렬 완료: %d → top-%d", len(chunks), len(ordered)
        )
        return ordered
