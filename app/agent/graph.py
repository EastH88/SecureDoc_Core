"""LangGraph 에이전트 그래프 정의 - 3단계 RAG 파이프라인 오케스트레이션"""

import logging
import uuid

from langgraph.graph import END, StateGraph

from app.config import settings
from app.models import AgentState, Citation, QueryResponse
from app.agent.nodes import (
    analyze_query,
    caption_images,
    generate_answer,
    retrieve_chunks,
    should_retry,
    verify_answer,
)

logger = logging.getLogger(__name__)

_compiled_graph = None
_tracer = None


def build_graph():
    """LangGraph StateGraph 구성 및 컴파일

    analyze_query → retrieve_chunks → caption_images
      → generate_answer → verify_answer
      → (should_retry) → retrieve_chunks (retry) 또는 END (done)
    """
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("caption_images", caption_images)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)

    # 엣지 연결
    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "caption_images")
    graph.add_edge("caption_images", "generate_answer")
    graph.add_edge("generate_answer", "verify_answer")

    # 조건부 엣지: 검증 실패 시 재검색
    graph.add_conditional_edges(
        "verify_answer",
        should_retry,
        {"retry": "retrieve_chunks", "done": END},
    )

    compiled = graph.compile()
    logger.info("LangGraph 에이전트 그래프 컴파일 완료")
    return compiled


def init_graph(tracer=None):
    """그래프 초기화"""
    global _compiled_graph, _tracer
    _compiled_graph = build_graph()
    _tracer = tracer
    return _compiled_graph


async def run_query(
    question: str, doc_id: str | None = None
) -> QueryResponse:
    """질의 실행 → QueryResponse 반환"""
    if _compiled_graph is None:
        raise RuntimeError("그래프가 초기화되지 않았습니다. init_graph()를 먼저 호출하세요.")

    trace_id = None
    if _tracer:
        trace_id = _tracer.start_trace(
            "query", {"question": question, "doc_id": doc_id}
        )

    initial_state: AgentState = {
        "question": question,
        "doc_id": doc_id,
        "query_type": "simple",
        "chunks": [],
        "visual_context": "",
        "answer": "",
        "citations": [],
        "faithfulness_score": 0.0,
        "retry_count": 0,
        "trace_id": trace_id,
    }

    logger.info("질의 실행 시작: '%s'", question[:80])
    result = _compiled_graph.invoke(initial_state)

    # Citation 객체 변환
    citations = []
    for c in result.get("citations", []):
        if isinstance(c, dict):
            citations.append(Citation(
                chunk_id=c.get("chunk_id", ""),
                page_num=c.get("page_num", 0),
                content_preview=c.get("content_preview", ""),
                relevance_score=c.get("relevance_score", 0.0),
            ))

    response = QueryResponse(
        answer=result.get("answer", ""),
        citations=citations,
        faithfulness_score=result.get("faithfulness_score", 0.0),
        query_type=result.get("query_type", "simple"),
        trace_id=trace_id,
    )

    if _tracer and trace_id:
        _tracer.end_trace(trace_id, {
            "answer": response.answer[:200],
            "faithfulness_score": response.faithfulness_score,
        })

    logger.info(
        "질의 완료: faithfulness=%.2f, citations=%d개",
        response.faithfulness_score, len(response.citations),
    )
    return response
