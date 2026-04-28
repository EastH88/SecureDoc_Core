"""VLM 클라이언트 - vLLM OpenAI 호환 API를 통한 Qwen2-VL-7B-AWQ 연동"""

import base64
import logging
from pathlib import Path

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class VLMClient:
    """vLLM에서 서빙되는 VLM(GPU 0)과 통신하는 이미지 캡셔닝 클라이언트"""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.vlm_base_url,
            api_key="EMPTY",
        )
        self._model = settings.VLM_MODEL
        logger.info("VLM 클라이언트 초기화: %s", settings.vlm_base_url)

    def caption_image(self, image_path: str) -> str:
        """이미지를 읽어 VLM으로 캡션(설명) 생성"""
        path = Path(image_path)
        if not path.exists():
            logger.warning("이미지 파일 없음: %s", image_path)
            return ""

        img_bytes = path.read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        suffix = path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
        media_type = mime.get(suffix, "image/png")

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "이 이미지의 내용을 한국어로 자세히 설명하세요. "
                                    "표, 그래프, 다이어그램이 포함되어 있다면 "
                                    "데이터와 구조를 구체적으로 서술하세요."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            caption = resp.choices[0].message.content.strip()
            logger.info("이미지 캡션 생성 완료: %s (%d자)", image_path, len(caption))
            return caption
        except Exception:
            logger.exception("VLM 이미지 캡션 생성 실패: %s", image_path)
            return ""

    def caption_images(self, image_paths: list[str]) -> list[str]:
        """여러 이미지를 순차적으로 캡션 생성"""
        return [self.caption_image(p) for p in image_paths]
