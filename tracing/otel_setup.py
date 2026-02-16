"""OTel SDK configuration with lazy initialization."""

from __future__ import annotations

import os
from typing import Optional

_tracer = None
_initialized = False


def _otel_available() -> bool:
    """Check whether the OpenTelemetry SDK is importable."""
    try:
        import opentelemetry  # noqa: F401
        return True
    except ImportError:
        return False


def init_tracing(service_name: str = "llm-eval") -> None:
    """Lazily initialize the OTel tracing pipeline.

    Configures:
    * ConsoleSpanExporter — always enabled.
    * OTLPSpanExporter   — enabled when OTEL_EXPORTER_OTLP_ENDPOINT is set.

    If the OpenTelemetry SDK is not installed the call is a silent no-op.
    """
    global _tracer, _initialized

    if _initialized:
        return

    _initialized = True

    if not _otel_available():
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Always add a console exporter
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Optionally add OTLP exporter when endpoint is configured
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """Return the configured tracer, or a no-op tracer / ``None``.

    If OTel is available but ``init_tracing`` has not been called yet, a no-op
    tracer from the default provider is returned.  If OTel is not installed,
    ``None`` is returned so callers can guard usage.
    """
    if _tracer is not None:
        return _tracer

    if _otel_available():
        from opentelemetry import trace
        return trace.get_tracer("llm-eval")

    return None
