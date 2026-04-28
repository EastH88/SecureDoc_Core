"""Langfuse 관측성 모듈 - 트레이스 수집 및 스팬 관리"""

import logging
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Langfuse 트레이스 수집기

    Langfuse가 설정되지 않은 경우 모든 메서드가 no-op으로 동작한다.
    """

    def __init__(self) -> None:
        self._client = None
        self._enabled = False

        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                self._enabled = True
                logger.info("Langfuse 연결 완료: %s", settings.LANGFUSE_HOST)
            except Exception:
                logger.warning("Langfuse 초기화 실패, 관측성 비활성화", exc_info=True)
        else:
            logger.info("Langfuse 미설정, 관측성 비활성화")

        self._traces: dict = {}

    def start_trace(self, name: str, input_data: dict) -> str:
        """새 트레이스 시작 → trace_id 반환"""
        trace_id = str(uuid.uuid4())
        if not self._enabled:
            return trace_id
        try:
            trace = self._client.trace(
                id=trace_id, name=name, input=input_data
            )
            self._traces[trace_id] = trace
        except Exception:
            logger.warning("트레이스 시작 실패", exc_info=True)
        return trace_id

    def add_span(
        self, trace_id: str, name: str, input_data: dict, output_data: dict
    ) -> None:
        """트레이스에 스팬 추가"""
        if not self._enabled:
            return
        try:
            trace = self._traces.get(trace_id)
            if trace:
                trace.span(name=name, input=input_data, output=output_data)
        except Exception:
            logger.warning("스팬 추가 실패: %s", name, exc_info=True)

    def end_trace(self, trace_id: str, output_data: dict) -> None:
        """트레이스 종료"""
        if not self._enabled:
            return
        try:
            trace = self._traces.pop(trace_id, None)
            if trace:
                trace.update(output=output_data)
                self._client.flush()
        except Exception:
            logger.warning("트레이스 종료 실패", exc_info=True)
