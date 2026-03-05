# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
# Service: Worker  (Celery — async ingestion pipeline)
# Build context: repo root  (docker compose build context: .)
#
# Heavier than the API image: includes PDF parsing (PyMuPDF) and full NLP stack
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System packages:
#   libpq-dev + gcc  → psycopg2 / SQLAlchemy sync driver
#   (PyMuPDF ≥ 1.23 ships pre-compiled wheels — no system MuPDF needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Shared internal package
COPY shared/ /build/shared/
RUN pip install --no-cache-dir /build/shared

# Worker-specific dependencies
COPY services/worker/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir -r /build/requirements.txt


# ── Stage 2: minimal runtime image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Celery: disable the deprecated prefork "solo" pool warning
    PYTHONWARNINGS="ignore::DeprecationWarning"

# Non-root worker user — Principle of Least Privilege
RUN useradd --no-create-home --no-log-init --shell /bin/false workeruser

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages \
                 /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Application code
COPY services/worker/worker/ /app/worker/

USER workeruser

# Celery worker
#   --queues=ingestion     → only consumes from the ingestion queue
#   --concurrency=4        → 4 prefork processes (tune with CELERY_CONCURRENCY)
#   --max-tasks-per-child  → guard against memory leaks in PDF parsing
CMD ["celery", \
     "-A", "worker.celery_app", \
     "worker", \
     "--loglevel=info", \
     "--queues=ingestion", \
     "--concurrency=4", \
     "--max-tasks-per-child=50"]
