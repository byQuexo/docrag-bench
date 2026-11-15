"""Logging setup: human-readable text locally, one JSON object per line in containers.

Why JSON in containers: log collectors (Loki, CloudWatch, the Kubernetes
logging stack) index fields, so ``level="ERROR" and route="/query"`` becomes a
query instead of a regex. Every line also carries the current OpenTelemetry
``trace_id``/``span_id``, so a log line jumps straight to its trace in Phoenix
or Langfuse.

Why the stdlib formatter instead of structlog / python-json-logger: all
existing code already logs through ``logging.getLogger(__name__)``; a ~40 line
Formatter keeps that and adds no dependency.

Choose with ``DOCRAG_LOG_FORMAT=json|text`` (the Docker image sets json).
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from opentelemetry import trace

LogFormat = Literal["text", "json"]

# Attributes every LogRecord has. Anything else was passed via `extra={...}` and
# goes into the JSON object as its own field.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "color_message",  # uvicorn's copy of msg with ANSI colour codes
}

# Third-party libraries that are chatty at INFO.
_NOISY = ("httpx", "httpx2", "httpcore", "huggingface_hub")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            # Same hex format the trace viewers show, so it can be pasted there.
            out["trace_id"] = format(ctx.trace_id, "032x")
            out["span_id"] = format(ctx.span_id, "016x")
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        # default=str: a stray non-JSON value (Path, datetime) must never make
        # logging itself raise.
        return json.dumps(out, default=str, ensure_ascii=False)


def setup_logging(fmt: LogFormat = "text", level: str = "INFO") -> None:
    """Configure the root logger once; uvicorn is told to use it (log_config=None)."""
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
