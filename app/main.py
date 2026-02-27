from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.documents import router as documents_router
from app.api.routes.search import router as search_router
from app.core.logging import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Antitrust RAG Core", version="0.1.0")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "http_request",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        extra={
            "event": "unhandled_exception",
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "Antitrust RAG Core",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest": "POST /documents/",
            "search": "POST /search/",
        },
    }


app.include_router(documents_router)
app.include_router(search_router)
