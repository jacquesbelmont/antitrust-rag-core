"""
Structured JSON logging with per-request correlation IDs.

Architecture
------------
* A ``contextvars.ContextVar`` stores the current correlation ID so that every
  log statement within an async request handler automatically includes it,
  regardless of whether the logger is called from application or infrastructure
  code — no need to pass the ID explicitly through every function call.
* The ``JsonFormatter`` reads the context variable on every ``format()`` call,
  so the ID is always current even across ``await`` boundaries.
* ``configure_logging()`` is idempotent and safe to call from the FastAPI
  lifespan hook.

Standard fields emitted on every record
----------------------------------------
timestamp   ISO-8601 UTC
level       DEBUG / INFO / WARNING / ERROR / CRITICAL
logger      Python logger name
message     Human-readable message
service     "api" (constant)

Optional fields (present when set)
-----------------------------------
correlation_id  UUID from X-Correlation-ID request header (or generated)
method          HTTP method (GET, POST, …)
path            URL path
status_code     HTTP response status
elapsed_ms      Request duration
document_id     When processing a specific document
error_type      Exception class name
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone

# Module-level ContextVar: set per request in RequestContextMiddleware.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "api",
        }

        cid = correlation_id_var.get("")
        if cid:
            payload["correlation_id"] = cid

        for field in (
            "method",
            "path",
            "status_code",
            "elapsed_ms",
            "document_id",
            "task_id",
            "event",
            "error_type",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exc_info"] = traceback.format_exception(*record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """
    Replace root logger handlers with a single JSON → stdout handler.
    Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
