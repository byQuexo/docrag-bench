"""JSON log lines: fields, extras, exceptions, and the trace id inside a span."""

import json
import logging
import sys

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from docrag.shared import tracing
from docrag.shared.logs import JsonFormatter


def _record(msg: str = "hello %s", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        "docrag.x", logging.WARNING, "f.py", 1, msg, ("you",), None
    )
    record.__dict__.update(extra)
    return record


def test_json_line_has_core_fields_and_extras() -> None:
    out = json.loads(JsonFormatter().format(_record(route="/query", status=200)))
    assert out["level"] == "WARNING"
    assert out["logger"] == "docrag.x"
    assert out["msg"] == "hello you"
    assert out["route"] == "/query" and out["status"] == 200
    assert out["ts"].endswith("+00:00")
    assert "trace_id" not in out  # no active span


def test_json_line_carries_trace_id_inside_a_span(
    spans: InMemorySpanExporter,
) -> None:
    with tracing.root_span("t") as span:
        out = json.loads(JsonFormatter().format(_record()))
    assert out["trace_id"] == format(span.get_span_context().trace_id, "032x")
    assert len(out["span_id"]) == 16


def test_exceptions_and_odd_values_never_break_logging() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            "x", logging.ERROR, "f.py", 1, "failed", (), sys.exc_info()
        )
    record.__dict__["path"] = object()  # not JSON-serializable
    out = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in out["exc_info"]
    assert out["path"].startswith("<object")


def test_uvicorn_color_message_is_dropped() -> None:
    out = json.loads(JsonFormatter().format(_record(color_message="\x1b[1mx\x1b[0m")))
    assert "color_message" not in out
