"""슬라이딩 윈도우 청커 - 문서 블록을 고정 크기 청크로 분할"""

import logging

from app.config import settings
from app.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


class SlidingWindowChunker:
    """슬라이딩 윈도우 방식으로 텍스트를 청크로 분할한다.

    - size=500 글자, overlap=50 글자
    - figure 블록은 이미지 참조만 포함하는 단일 청크 생성
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk(self, blocks: list[dict], doc_id: str) -> list[Chunk]:
        """텍스트/테이블/이미지 블록을 청크 리스트로 변환"""
        chunks: list[Chunk] = []
        idx = 0

        for block in blocks:
            block_type = block.get("type", "text")

            # figure 블록 → 단일 청크 (이미지 경로 참조)
            if block_type == "figure":
                chunks.append(Chunk(
                    content=f"[Figure: {block.get('image_path', '')}]",
                    metadata=ChunkMetadata(
                        chunk_id=f"{doc_id}_chunk_{idx}",
                        doc_id=doc_id,
                        page_num=block["page_num"],
                        bbox=block.get("bbox"),
                        chunk_type="figure",
                        image_path=block.get("image_path"),
                    ),
                ))
                idx += 1
                continue

            # 텍스트/테이블 블록 → 슬라이딩 윈도우 분할
            text = block.get("content", "")
            if not text.strip():
                continue

            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                if chunk_text.strip():
                    chunks.append(Chunk(
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            chunk_id=f"{doc_id}_chunk_{idx}",
                            doc_id=doc_id,
                            page_num=block["page_num"],
                            bbox=block.get("bbox"),
                            chunk_type=block_type,
                        ),
                    ))
                    idx += 1

                if end >= len(text):
                    break
                start = end - self.chunk_overlap

        logger.info("문서 %s: %d개 청크 생성", doc_id, len(chunks))
        return chunks
