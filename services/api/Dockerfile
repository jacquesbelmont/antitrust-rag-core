# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# Service: API  (FastAPI — synchronous REST gateway)
# Build context: repo root  (docker compose build context: .)
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System packages required by asyncpg (Postgres C extension) and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the shared internal package first
COPY shared/ /build/shared/
RUN pip install --no-cache-dir /build/shared

# Install service-specific dependencies
COPY services/api/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt


# ── Stage 2: minimal runtime image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user — Principle of Least Privilege
RUN useradd --no-create-home --no-log-init --shell /bin/false appuser

WORKDIR /app

# Pull installed packages from the builder stage (no compiler tools in runtime)
COPY --from=deps /usr/local/lib/python3.11/site-packages \
                 /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Application code
COPY services/api/app/ /app/app/

USER appuser

EXPOSE 8000

# 4 Uvicorn workers — tune via UVICORN_WORKERS env var at runtime if needed
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--access-log"]
