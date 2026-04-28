"""LangGraph 에이전트 프롬프트 템플릿"""

QUERY_CLASSIFICATION_PROMPT = """\
다음 질문을 분석하여 타입을 분류하세요.

- simple: 단일 문단에서 답을 찾을 수 있는 단순 사실 질문
- multi_hop: 여러 문단의 정보를 종합해야 하는 복합 질문
- visual: 그래프, 표, 이미지 등 시각 자료 참조가 필요한 질문

질문: {question}

반드시 "simple", "multi_hop", "visual" 중 하나만 출력하세요."""

ANSWER_GENERATION_PROMPT = """\
다음 컨텍스트를 참고하여 질문에 정확하게 답변하세요.
반드시 근거가 되는 청크를 citation으로 포함해야 합니다.
컨텍스트에 없는 내용은 답변에 포함하지 마세요.

[컨텍스트]
{context}
{visual_context}

[질문]
{question}

아래 JSON 형식으로만 응답하세요:
{{"answer": "답변 내용", "citations": [{{"chunk_id": "청크ID", "page_num": 페이지번호, "content_preview": "근거가 되는 텍스트 일부", "relevance_score": 0.0~1.0}}]}}"""

FAITHFULNESS_VERIFICATION_PROMPT = """\
다음 답변이 주어진 컨텍스트에 얼마나 충실한지 평가하세요.

평가 기준:
- 답변의 모든 주장이 컨텍스트에 근거하는가?
- 컨텍스트에 없는 내용이 포함되어 있지 않은가?
- 답변이 질문에 적절히 대응하는가?

[컨텍스트]
{context}

[질문]
{question}

[답변]
{answer}

0.0부터 1.0 사이의 숫자만 출력하세요. (1.0 = 완전히 충실, 0.0 = 전혀 충실하지 않음)"""

IMAGE_CAPTION_PROMPT = """\
이 이미지의 내용을 한국어로 자세히 설명하세요.
표, 그래프, 다이어그램이 포함되어 있다면 데이터와 구조를 구체적으로 서술하세요.
수치가 포함된 경우 정확한 값을 명시하세요."""
