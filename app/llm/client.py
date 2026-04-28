"""LLM 클라이언트 - vLLM OpenAI 호환 API를 통한 Qwen2.5-14B-AWQ 연동"""

import json
import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """vLLM에서 서빙되는 LLM(GPU 1)과 통신하는 클라이언트"""

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
    ) -> str:
        """LLM에 프롬프트를 전송하고 응답을 반환"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
        """검색 컨텍스트 기반 답변 + citation 생성"""
        vis_section = ""
        if visual_context:
            vis_section = f"\n\n[이미지 설명]\n{visual_context}"

        prompt = (
            "다음 컨텍스트를 참고하여 질문에 정확하게 답변하세요.\n"
            "반드시 근거가 되는 청크를 citation으로 포함하세요.\n\n"
            f"[컨텍스트]\n{context}{vis_section}\n\n"
            f"[질문]\n{question}\n\n"
            "아래 JSON 형식으로만 응답하세요:\n"
            '{"answer": "답변 내용", "citations": [{"chunk_id": "청크ID", '
            '"page_num": 페이지번호, "content_preview": "근거 미리보기", '
            '"relevance_score": 0.0~1.0}]}'
        )
        raw = self.generate(prompt, temperature=0.1, max_tokens=2048)

        try:
            # JSON 블록 추출
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("LLM 응답 JSON 파싱 실패, 원문 응답 사용")
            return {"answer": raw, "citations": []}

    def verify_faithfulness(
        self, question: str, answer: str, context: str
    ) -> float:
        """답변의 faithfulness 점수 검증 (0.0 ~ 1.0)"""
        prompt = (
            "다음 답변이 주어진 컨텍스트에 얼마나 충실한지 평가하세요.\n"
            "컨텍스트에 없는 내용이 포함되면 낮은 점수를 부여합니다.\n\n"
            f"[컨텍스트]\n{context}\n\n"
            f"[질문]\n{question}\n\n"
            f"[답변]\n{answer}\n\n"
            "0.0부터 1.0 사이의 숫자만 출력하세요."
        )
        raw = self.generate(prompt, temperature=0.0, max_tokens=10)
        try:
            score = float(raw.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            logger.warning("Faithfulness 점수 파싱 실패: '%s'", raw)
            return 0.5
