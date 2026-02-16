"""Tests for the tracing package.

These tests must pass **without** OTel or aiokafka installed — all external
dependencies are mocked or handled via ImportError guards.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from unittest import mock


class TestTraceDecorator(unittest.TestCase):
    """Tests for tracing.span_decorators.trace_eval."""

    def _force_reimport(self):
        """Remove cached tracing modules so they pick up patched state."""
        for mod_name in list(sys.modules):
            if mod_name.startswith("tracing"):
                del sys.modules[mod_name]

    # ---- works without OTel ------------------------------------------------

    def test_sync_works_without_otel(self):
        """Decorated sync function runs normally when OTel is absent."""
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            def add(a, b):
                return a + b

            self.assertEqual(add(1, 2), 3)

    def test_async_works_without_otel(self):
        """Decorated async function runs normally when OTel is absent."""
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            async def add_async(a, b):
                return a + b

            result = asyncio.get_event_loop().run_until_complete(add_async(1, 2))
            self.assertEqual(result, 3)

    # ---- preserves function name -------------------------------------------

    def test_preserves_sync_function_name(self):
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            def my_eval_func():
                pass

            self.assertEqual(my_eval_func.__name__, "my_eval_func")

    def test_preserves_async_function_name(self):
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            async def my_async_eval():
                pass

            self.assertEqual(my_async_eval.__name__, "my_async_eval")

    # ---- propagates exceptions ---------------------------------------------

    def test_sync_propagates_exception(self):
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            def boom():
                raise ValueError("kaboom")

            with self.assertRaises(ValueError):
                boom()

    def test_async_propagates_exception(self):
        with mock.patch.dict(sys.modules, {"opentelemetry": None}):
            self._force_reimport()
            from tracing.span_decorators import trace_eval

            @trace_eval
            async def boom_async():
                raise ValueError("kaboom")

            with self.assertRaises(ValueError):
                asyncio.get_event_loop().run_until_complete(boom_async())


class TestKafkaTraceExporter(unittest.TestCase):
    """Tests for tracing.kafka_exporter.KafkaTraceExporter."""

    def test_export_without_producer_does_not_raise(self):
        """Calling export_span when no producer is available must not raise."""
        # Ensure aiokafka is treated as missing
        with mock.patch.dict(sys.modules, {"aiokafka": None}):
            for mod_name in list(sys.modules):
                if mod_name.startswith("tracing"):
                    del sys.modules[mod_name]

            from tracing.kafka_exporter import KafkaTraceExporter

            exporter = KafkaTraceExporter()
            # producer is None because aiokafka is unavailable
            self.assertIsNone(exporter._producer)

            # export_span should complete silently
            asyncio.get_event_loop().run_until_complete(
                exporter.export_span(
                    trace_id="abc123",
                    span_id="span456",
                    operation="test_op",
                    service="test_svc",
                    duration_ms=42.0,
                )
            )

    def test_start_without_aiokafka_does_not_raise(self):
        """Calling start() when aiokafka is missing must not raise."""
        with mock.patch.dict(sys.modules, {"aiokafka": None}):
            for mod_name in list(sys.modules):
                if mod_name.startswith("tracing"):
                    del sys.modules[mod_name]

            from tracing.kafka_exporter import KafkaTraceExporter

            exporter = KafkaTraceExporter()
            asyncio.get_event_loop().run_until_complete(exporter.start())
            self.assertIsNone(exporter._producer)


if __name__ == "__main__":
    unittest.main()
