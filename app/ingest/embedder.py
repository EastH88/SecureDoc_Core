"""BGE-M3 임베딩 + Kiwi 한국어 형태소 분석"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """BGE-M3 기반 Dense 임베딩(1024차원) 및 Kiwi 형태소 토크나이저

    모델은 최초 호출 시 지연 로딩(lazy loading)된다.
    """

    _instance: "Embedder | None" = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model = None
        self._kiwi = None
        self._initialized = True

    # ── 모델 지연 로딩 ──────────────────────────────────────
    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info("임베딩 모델 로딩: %s", settings.EMBEDDING_MODEL)
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("임베딩 모델 로딩 완료")

    def _load_kiwi(self) -> None:
        if self._kiwi is not None:
            return
        from kiwipiepy import Kiwi

        logger.info("Kiwi 형태소 분석기 초기화")
        self._kiwi = Kiwi()

    # ── 임베딩 ──────────────────────────────────────────────
    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 Dense 임베딩 벡터(1024차원)로 변환"""
        self._load_model()
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리 임베딩"""
        return self.embed([text])[0]

    # ── 한국어 형태소 토큰화 ────────────────────────────────
    def tokenize_korean(self, text: str) -> list[str]:
        """Kiwi 형태소 분석기로 한국어 텍스트를 토큰화 (BM25용)"""
        self._load_kiwi()
        tokens = self._kiwi.tokenize(text)
        # 의미 있는 품사만 추출 (명사, 동사, 형용사, 부사 등)
        meaningful_tags = {"NNG", "NNP", "NNB", "VV", "VA", "MAG", "SL", "SN"}
        return [
            token.form
            for token in tokens
            if token.tag in meaningful_tags and len(token.form) > 1
        ]
