"""
Request context middleware.

Responsibilities
----------------
1. **Correlation ID injection** — reads ``X-Correlation-ID`` from the request
   header (or generates a fresh UUID) and stores it in ``correlation_id_var``
   (a ``contextvars.ContextVar``).  Every subsequent log statement within the
   same async task will automatically include this ID.

2. **Request / response structured logging** — logs method, path, status code,
   and elapsed time for every request.  No path parameters or query strings are
   logged to avoid inadvertently logging PII.

3. **Security headers** — adds a minimal set of HTTP security headers on every
   response.  These are not a substitute for a proper API gateway / WAF, but
   they provide baseline protection for direct callers.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging import correlation_id_var

logger = logging.getLogger(__name__)

_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Cache-Control": "no-store",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Set correlation ID in contextvar and log every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)

        start = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1_000, 1)
            correlation_id_var.reset(token)

            logger.info(
                "HTTP request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code if response is not None else 500,
                    "elapsed_ms": elapsed_ms,
                },
            )

        # Propagate the correlation ID back to the caller
        response.headers["X-Correlation-ID"] = correlation_id

        # Minimal security headers
        for header, value in _SECURITY_HEADERS.items():
            response.headers[header] = value

        return response
