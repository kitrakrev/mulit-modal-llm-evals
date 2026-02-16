"""Direct Kafka trace exporter (interim solution).

Uses *aiokafka* when available; degrades gracefully when the library is not
installed so that the rest of the tracing package remains functional.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_aiokafka_available = False
try:
    from aiokafka import AIOKafkaProducer  # type: ignore[import-untyped]
    _aiokafka_available = True
except ImportError:
    AIOKafkaProducer = None  # type: ignore[assignment,misc]


class KafkaTraceExporter:
    """Lightweight async Kafka exporter for trace spans.

    Parameters
    ----------
    bootstrap_servers:
        Comma-separated list of Kafka broker addresses.
    topic:
        Kafka topic to publish span data to (default ``traces.spans``).
    """

    DEFAULT_TOPIC = "traces.spans"

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = DEFAULT_TOPIC,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._producer: Optional[Any] = None

    async def start(self) -> None:
        """Create and start the underlying Kafka producer."""
        if not _aiokafka_available:
            logger.warning(
                "aiokafka is not installed — KafkaTraceExporter will be a no-op"
            )
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()

    async def stop(self) -> None:
        """Flush and stop the Kafka producer."""
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def export_span(
        self,
        trace_id: str,
        span_id: str,
        operation: str,
        service: str,
        parent_span_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: str = "ok",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Serialize and publish a single span record to Kafka.

        If the producer has not been started (e.g. aiokafka is missing) the
        call is silently ignored so that callers do not need to guard every
        invocation.
        """
        if self._producer is None:
            logger.debug("No Kafka producer available — dropping span %s", span_id)
            return

        payload: Dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "operation": operation,
            "service": service,
            "parent_span_id": parent_span_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "status": status,
            "attributes": attributes or {},
        }

        await self._producer.send_and_wait(self._topic, payload)
