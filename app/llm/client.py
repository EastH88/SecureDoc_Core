"""LLM 클라이언트 - vLLM/Ollama OpenAI 호환 API 연동"""

import json
import logging
import re

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"([01](?:\.\d+)?|0?\.\d+)")


class LLMClient:
    """OpenAI 호환 엔드포인트(vLLM, Ollama 등)를 통해 LLM에 질의"""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key="EMPTY",
        )
        self._model = settings.LLM_MODEL
        logger.info("LLM 클라이언트 초기화: %s", settings.llm_base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """LLM에 프롬프트를 전송하고 응답 텍스트를 반환"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception:
            logger.exception("LLM 생성 실패")
            return ""

    def classify_query(self, question: str) -> str:
        """질문 타입 분류 → simple / multi_hop / visual"""
        prompt = (
            "다음 질문을 분석하여 타입을 분류하세요.\n"
            "- simple: 단일 문단에서 답을 찾을 수 있는 단순 질문\n"
            "- multi_hop: 여러 문단의 정보를 종합해야 하는 복합 질문\n"
            "- visual: 그래프, 표, 이미지 등 시각 자료 참조가 필요한 질문\n\n"
            f"질문: {question}\n\n"
            '반드시 "simple", "multi_hop", "visual" 중 하나만 출력하세요.'
        )
        result = self.generate(prompt, temperature=0.0, max_tokens=20)
        result = result.lower().strip().strip('"').strip("'")
        if result in ("simple", "multi_hop", "visual"):
            return result
        return "simple"

    def generate_answer(
        self, question: str, context: str, visual_context: str = ""
    ) -> dict:
        """검색 컨텍스트 기반 답변 + citation 생성 (JSON 강제)"""
        vis_section = f"\n\n[이미지 설명]\n{visual_context}" if visual_context else ""

        prompt = (
            "당신은 사내 문서 Q&A 어시스턴트입니다. 아래 [컨텍스트]만 근거로 "
            "[질문]에 한국어로 답하세요. 컨텍스트에 없는 사실은 추측하지 마세요.\n\n"
            "각 청크는 `[청크 <chunk_id>] (p.<page_num>) <내용>` 형식으로 주어집니다.\n"
            "답변에 사용한 청크는 citations 배열에 빠짐없이 넣으세요.\n"
            "- chunk_id, page_num은 컨텍스트에서 보이는 값을 그대로 복사할 것\n"
            "- content_preview는 해당 청크의 핵심 문장 1~2개를 그대로 발췌\n"
            "- relevance_score는 답변 근거로서의 강도(0.0~1.0)\n"
            "컨텍스트로 답할 수 없으면 answer에 '문서에서 관련 정보를 찾지 못했습니다.'를 쓰고 "
            "citations는 빈 배열로 두세요.\n\n"
            f"[컨텍스트]\n{context}{vis_section}\n\n"
            f"[질문]\n{question}\n\n"
            "오직 다음 JSON 객체 하나만 출력하세요(설명·코드블록 금지):\n"
            '{"answer": "...", "citations": [{"chunk_id": "...", '
            '"page_num": 0, "content_preview": "...", "relevance_score": 0.0}]}'
        )
        raw = self.generate(
            prompt, temperature=0.1, max_tokens=2048, json_mode=True
        )

        return self._parse_answer_json(raw)

    @staticmethod
    def _parse_answer_json(raw: str) -> dict:
        """LLM 응답에서 JSON 객체를 견고하게 추출"""
        if not raw:
            return {"answer": "", "citations": []}

        text = raw
        # 코드블록 제거
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]

        # 가장 바깥 { ... } 추출
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        try:
            data = json.loads(text.strip())
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            data.setdefault("answer", "")
            cites = data.get("citations") or []
            if not isinstance(cites, list):
                cites = []
            data["citations"] = cites
            return data
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM 응답 JSON 파싱 실패, 원문 응답 사용")
            return {"answer": raw, "citations": []}

    def verify_faithfulness(
        self, question: str, answer: str, context: str
    ) -> float:
        """답변의 faithfulness 점수 검증 (0.0 ~ 1.0). 파싱 실패 시 -1.0"""
        prompt = (
            "다음 답변이 [컨텍스트]에 얼마나 충실한지 0.0부터 1.0 사이 숫자 하나로만 평가하세요.\n"
            "- 1.0: 모든 주장이 컨텍스트에 직접 근거\n"
            "- 0.5: 일부 근거 있음\n"
            "- 0.0: 컨텍스트와 무관하거나 환각\n"
            "다른 텍스트는 절대 출력하지 마세요. 숫자 하나만.\n\n"
            f"[컨텍스트]\n{context}\n\n"
            f"[질문]\n{question}\n\n"
            f"[답변]\n{answer}\n\n점수:"
        )
        raw = self.generate(prompt, temperature=0.0, max_tokens=10)
        score = self._parse_score(raw)
        if score is None:
            logger.warning("Faithfulness 점수 파싱 실패: %r", raw)
            return -1.0
        return score

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        """응답 문자열에서 0.0~1.0 사이 숫자 하나를 추출"""
        if not raw:
            return None
        m = _SCORE_RE.search(raw)
        if not m:
            return None
        try:
            value = float(m.group(1))
        except ValueError:
            return None
        return max(0.0, min(1.0, value))
