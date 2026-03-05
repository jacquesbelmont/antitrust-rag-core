"""
Structured JSON logging for the worker service.

Every log record is serialised as a single-line JSON object so that log
aggregators (Loki, CloudWatch, Datadog) can parse fields without fragile
regex patterns.

Standard fields emitted on every record
----------------------------------------
timestamp   ISO-8601 UTC
level       DEBUG / INFO / WARNING / ERROR / CRITICAL
logger      Python logger name
message     Human-readable message
service     "worker" (constant)

Optional fields (present when set via ``extra=``)
--------------------------------------------------
document_id, task_id, event, error_type, exc_info
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": "worker",
        }

        # Propagate structured extra fields
        for field in ("document_id", "task_id", "event", "error_type"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exc_info"] = traceback.format_exception(*record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """
    Replace the root logger's handlers with a single JSON → stdout handler.
    Safe to call multiple times (idempotent).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any pre-existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
