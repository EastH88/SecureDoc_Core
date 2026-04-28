"""LangGraph 에이전트 노드 함수 - 질의 분석, 검색, 캡셔닝, 생성, 검증"""

import logging

from app.models import AgentState

logger = logging.getLogger(__name__)

# 서비스 인스턴스 (main.py에서 초기화 후 주입)
_llm_client = None
_vlm_client = None
_hybrid_retriever = None
_tracer = None


def init_services(llm_client, vlm_client, hybrid_retriever, tracer) -> None:
    """노드에서 사용할 서비스 인스턴스 주입"""
    global _llm_client, _vlm_client, _hybrid_retriever, _tracer
    _llm_client = llm_client
    _vlm_client = vlm_client
    _hybrid_retriever = hybrid_retriever
    _tracer = tracer


def analyze_query(state: AgentState) -> AgentState:
    """질문 타입 분류 (simple / multi_hop / visual)"""
    question = state["question"]
    logger.info("질문 분류 시작: '%s'", question[:50])

    query_type = _llm_client.classify_query(question)
    logger.info("질문 타입: %s", query_type)

    if _tracer and state.get("trace_id"):
        _tracer.add_span(
            state["trace_id"], "analyze_query",
            {"question": question}, {"query_type": query_type},
        )

    return {**state, "query_type": query_type}


def retrieve_chunks(state: AgentState) -> AgentState:
    """하이브리드 검색으로 관련 청크 top-5 검색"""
    question = state["question"]
    doc_id = state.get("doc_id")
    logger.info("하이브리드 검색 시작 (doc_id=%s)", doc_id)

    chunks = _hybrid_retriever.search(query=question, doc_id=doc_id)
    chunk_dicts = [
        {
            "chunk_id": c.metadata.chunk_id,
            "content": c.content,
            "page_num": c.metadata.page_num,
            "chunk_type": c.metadata.chunk_type,
            "image_path": c.metadata.image_path,
        }
        for c in chunks
    ]
    logger.info("검색 결과: %d개 청크", len(chunk_dicts))

    if _tracer and state.get("trace_id"):
        _tracer.add_span(
            state["trace_id"], "retrieve_chunks",
            {"question": question, "doc_id": doc_id},
            {"chunk_count": len(chunk_dicts)},
        )

    return {**state, "chunks": chunk_dicts}


def caption_images(state: AgentState) -> AgentState:
    """figure 청크가 있으면 VLM으로 이미지 캡셔닝"""
    chunks = state.get("chunks", [])
    image_paths = [
        c["image_path"]
        for c in chunks
        if c.get("chunk_type") == "figure" and c.get("image_path")
    ]

    if not image_paths:
        return {**state, "visual_context": ""}

    logger.info("이미지 캡셔닝 시작: %d개", len(image_paths))
    captions = _vlm_client.caption_images(image_paths)
    visual_context = "\n\n".join(
        f"[이미지 {i+1}] {cap}" for i, cap in enumerate(captions) if cap
    )

    if _tracer and state.get("trace_id"):
        _tracer.add_span(
            state["trace_id"], "caption_images",
            {"image_count": len(image_paths)},
            {"visual_context_length": len(visual_context)},
        )

    return {**state, "visual_context": visual_context}


def generate_answer(state: AgentState) -> AgentState:
    """검색 컨텍스트 + 비주얼 컨텍스트로 답변 생성"""
    question = state["question"]
    chunks = state.get("chunks", [])
    visual_context = state.get("visual_context", "")

    context = "\n\n".join(
        f"[청크 {c['chunk_id']}] (p.{c['page_num']})\n{c['content']}"
        for c in chunks
    )

    logger.info("답변 생성 시작")
    result = _llm_client.generate_answer(question, context, visual_context)

    answer = result.get("answer", "")
    citations = result.get("citations", [])

    if _tracer and state.get("trace_id"):
        _tracer.add_span(
            state["trace_id"], "generate_answer",
            {"question": question, "context_length": len(context)},
            {"answer_length": len(answer), "citation_count": len(citations)},
        )

    return {**state, "answer": answer, "citations": citations}


def verify_answer(state: AgentState) -> AgentState:
    """답변의 faithfulness 점수 검증"""
    question = state["question"]
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])

    context = "\n\n".join(c["content"] for c in chunks)

    logger.info("Faithfulness 검증 시작")
    score = _llm_client.verify_faithfulness(question, answer, context)
    retry_count = state.get("retry_count", 0)

    logger.info("Faithfulness 점수: %.2f (시도 %d회)", score, retry_count + 1)

    if _tracer and state.get("trace_id"):
        _tracer.add_span(
            state["trace_id"], "verify_answer",
            {"answer": answer[:200]},
            {"faithfulness_score": score, "retry_count": retry_count},
        )

    return {
        **state,
        "faithfulness_score": score,
        "retry_count": retry_count + 1,
    }


def should_retry(state: AgentState) -> str:
    """검증 실패 시 재시도 여부 결정"""
    from app.config import settings

    score = state.get("faithfulness_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if score < settings.FAITHFULNESS_THRESHOLD and retry_count < settings.MAX_RETRIES:
        logger.info("재검색 루프 진입 (score=%.2f < %.2f)", score, settings.FAITHFULNESS_THRESHOLD)
        return "retry"
    return "done"
