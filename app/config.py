"""SecureDoc Core 설정 모듈 - 환경변수 기반 중앙 설정"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FastAPI 서버
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # VLM (GPU 0) - 이미지 캡셔닝
    VLM_HOST: str = "localhost"
    VLM_PORT: int = 8001
    VLM_MODEL: str = "Qwen/Qwen2-VL-7B-Instruct-AWQ"

    # LLM (GPU 1) - 추론 및 검증
    LLM_HOST: str = "localhost"
    LLM_PORT: int = 8002
    LLM_MODEL: str = "Qwen/Qwen2.5-14B-Instruct-AWQ"

    # 임베딩 모델
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # ChromaDB 벡터 저장소
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "securedoc"

    # BM25 역색인 저장소
    BM25_INDEX_DIR: str = "./data/bm25_index"

    # 청킹 설정
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # 검색 설정
    TOP_K: int = 5
    RRF_K: int = 60
    FAITHFULNESS_THRESHOLD: float = 0.7
    MAX_RETRIES: int = 2

    # Langfuse 관측성
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    # 파일 저장 경로
    UPLOAD_DIR: str = "./data/uploads"
    IMAGE_DIR: str = "./data/images"

    @property
    def vlm_base_url(self) -> str:
        return f"http://{self.VLM_HOST}:{self.VLM_PORT}/v1"

    @property
    def llm_base_url(self) -> str:
        return f"http://{self.LLM_HOST}:{self.LLM_PORT}/v1"

    def ensure_dirs(self) -> None:
        """필요한 디렉토리 생성"""
        for d in [self.CHROMA_PERSIST_DIR, self.BM25_INDEX_DIR,
                  self.UPLOAD_DIR, self.IMAGE_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
