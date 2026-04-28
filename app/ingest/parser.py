"""PDF 파서 - PyMuPDF로 텍스트/이미지 추출, pdfplumber로 표 구조 추출"""

import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.config import settings

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF 문서에서 텍스트, 이미지, 표를 추출하는 파서"""

    def __init__(self) -> None:
        self.image_dir = Path(settings.IMAGE_DIR)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def parse(
        self, file_path: str, doc_id: str
    ) -> tuple[list[dict], list[dict]]:
        """PDF를 파싱하여 텍스트 블록과 이미지 블록을 반환한다.

        Returns:
            (text_blocks, image_blocks)
        """
        text_blocks = self._extract_text_and_tables(file_path, doc_id)
        image_blocks = self._extract_images(file_path, doc_id)
        logger.info(
            "문서 %s 파싱 완료: 텍스트 %d개, 이미지 %d개",
            doc_id, len(text_blocks), len(image_blocks),
        )
        return text_blocks, image_blocks

    def get_page_count(self, file_path: str) -> int:
        """PDF 페이지 수 반환"""
        doc = fitz.open(file_path)
        count = len(doc)
        doc.close()
        return count

    # ── 텍스트 + 표 추출 ───────────────────────────────────
    def _extract_text_and_tables(
        self, file_path: str, doc_id: str
    ) -> list[dict]:
        blocks: list[dict] = []

        # pdfplumber로 표 추출
        table_regions = self._extract_tables(file_path, doc_id)

        # PyMuPDF로 텍스트 블록 추출
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # 텍스트 블록만
                    continue
                bbox = block.get("bbox", (0, 0, 0, 0))
                lines = block.get("lines", [])
                text = ""
                for line in lines:
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                    text += "\n"
                text = text.strip()
                if not text:
                    continue
                blocks.append({
                    "content": text,
                    "page_num": page_num,
                    "bbox": list(bbox),
                    "type": "text",
                })
        doc.close()

        # 표 블록 추가
        blocks.extend(table_regions)
        # 페이지 순서로 정렬
        blocks.sort(key=lambda b: (b["page_num"], b.get("bbox", [0])[0]))
        return blocks

    def _extract_tables(self, file_path: str, doc_id: str) -> list[dict]:
        """pdfplumber로 표를 마크다운 형식으로 추출"""
        table_blocks: list[dict] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:
                            continue
                        md = self._table_to_markdown(table)
                        if md.strip():
                            table_blocks.append({
                                "content": md,
                                "page_num": page_num,
                                "bbox": [0, 0, 0, 0],
                                "type": "table",
                            })
        except Exception:
            logger.warning("표 추출 실패: %s", file_path, exc_info=True)
        return table_blocks

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """2D 리스트 → 마크다운 표 변환"""
        if not table:
            return ""
        rows = []
        for i, row in enumerate(table):
            cells = [str(c).replace("\n", " ").strip() if c else "" for c in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)

    # ── 이미지 추출 ─────────────────────────────────────────
    def _extract_images(self, file_path: str, doc_id: str) -> list[dict]:
        """PyMuPDF로 이미지를 추출하여 PNG로 저장"""
        image_blocks: list[dict] = []
        doc = fitz.open(file_path)

        for page_num, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK → RGB 변환
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_name = f"{doc_id}_p{page_num}_img{img_idx}.png"
                    img_path = self.image_dir / img_name
                    pix.save(str(img_path))

                    image_blocks.append({
                        "image_path": str(img_path),
                        "page_num": page_num,
                        "bbox": [0, 0, pix.width, pix.height],
                        "type": "figure",
                    })
                except Exception:
                    logger.warning(
                        "이미지 추출 실패: page=%d, xref=%d",
                        page_num, xref, exc_info=True,
                    )
        doc.close()
        return image_blocks
