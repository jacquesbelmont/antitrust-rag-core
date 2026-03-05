# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical Legal RAG — Developer Makefile
#
# Prerequisites: Docker, Docker Compose v2, Python ≥ 3.11 (for local test/lint)
#
# Quick start:
#   cp .env.example .env   # fill in secrets
#   make build             # build images
#   make up                # start all services
#   make migrate           # run DB migrations
#   make logs              # tail all logs
# ──────────────────────────────────────────────────────────────────────────────

# Load .env into make variables (does not export to shell by default)
-include .env

# Provide sensible defaults so `make` works even without a .env file
POSTGRES_USER     ?= legalrag
POSTGRES_DB       ?= legalrag
POSTGRES_PASSWORD ?= changeme
API_PORT          ?= 8000
FLOWER_PORT       ?= 5555

.DEFAULT_GOAL := help

.PHONY: help up down restart build build-no-cache \
        logs logs-api logs-worker logs-flower ps \
        migrate shell-api shell-worker shell-db \
        test test-unit lint typecheck \
        clean clean-volumes


# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  Legal RAG — available targets"
	@echo ""
	@echo "  Services"
	@echo "    make up              Start all services in the background"
	@echo "    make down            Stop and remove containers (keeps volumes)"
	@echo "    make restart         Restart all services"
	@echo "    make build           Build Docker images"
	@echo "    make build-no-cache  Rebuild images from scratch"
	@echo "    make ps              Show running services and ports"
	@echo ""
	@echo "  Logs"
	@echo "    make logs            Tail all service logs"
	@echo "    make logs-api        Tail API service logs"
	@echo "    make logs-worker     Tail Celery worker logs"
	@echo "    make logs-flower     Tail Flower logs"
	@echo ""
	@echo "  Database"
	@echo "    make migrate         Run Alembic migrations (upgrade head)"
	@echo "    make shell-db        Open psql inside the postgres container"
	@echo ""
	@echo "  Shells"
	@echo "    make shell-api       sh session inside the API container"
	@echo "    make shell-worker    sh session inside the worker container"
	@echo ""
	@echo "  Tests & quality"
	@echo "    make test            Run all tests (unit + integration)"
	@echo "    make test-unit       Run unit tests only (no infrastructure)"
	@echo "    make lint            Run ruff linter"
	@echo "    make typecheck       Run mypy type checker"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean           Remove containers and images"
	@echo "    make clean-volumes   Remove containers, images AND data volumes (!)"
	@echo ""


# ── Services ──────────────────────────────────────────────────────────────────

up:
	docker compose up -d
	@echo ""
	@echo "  Services started:"
	@echo "    API    → http://localhost:$(API_PORT)/docs"
	@echo "    Flower → http://localhost:$(FLOWER_PORT)"
	@echo ""

down:
	docker compose down

restart:
	docker compose restart

build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

ps:
	docker compose ps


# ── Logs ──────────────────────────────────────────────────────────────────────

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

logs-flower:
	docker compose logs -f flower


# ── Database / Migrations ─────────────────────────────────────────────────────
#
# Alembic requires ALEMBIC_DATABASE_URL (postgresql+psycopg2://...).
# We run it inside the API container so it can reach postgres via the internal
# Docker network (postgres is not published to the host in production).
#
# The infra/ directory is bind-mounted at /infra inside the container because
# the API Dockerfile does not COPY infra/ into the image (keeping images small).

migrate:
	@echo "Running Alembic migrations (upgrade head)..."
	docker compose run --rm \
		-v "$(CURDIR)/infra:/infra:ro" \
		-e "ALEMBIC_DATABASE_URL=postgresql+psycopg2://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@postgres:5432/$(POSTGRES_DB)" \
		--entrypoint="" \
		api \
		alembic -c /infra/migrations/alembic.ini upgrade head

migrate-history:
	@echo "Alembic migration history:"
	docker compose run --rm \
		-v "$(CURDIR)/infra:/infra:ro" \
		-e "ALEMBIC_DATABASE_URL=postgresql+psycopg2://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@postgres:5432/$(POSTGRES_DB)" \
		--entrypoint="" \
		api \
		alembic -c /infra/migrations/alembic.ini history --verbose

shell-db:
	docker compose exec postgres psql -U $(POSTGRES_USER) $(POSTGRES_DB)


# ── Development shells ────────────────────────────────────────────────────────

shell-api:
	docker compose exec api /bin/sh

shell-worker:
	docker compose exec worker /bin/sh


# ── Tests & quality ───────────────────────────────────────────────────────────
#
# Tests run on the host using the virtual environment.
# Activate your venv first:  source .venv/bin/activate
#
# Unit tests are isolated (no Docker required — uses mocks/in-memory fakes).
# Integration tests require all services to be running: make up && make migrate

test:
	pytest tests/ services/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

lint:
	ruff check services/ shared/ tests/

lint-fix:
	ruff check --fix services/ shared/ tests/

typecheck:
	mypy services/api/app services/worker/worker shared/legal_rag_shared \
		--ignore-missing-imports \
		--strict


# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	docker compose down --rmi local

clean-volumes:
	@echo "WARNING: This will permanently delete all PostgreSQL, Redis and Weaviate data."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down -v --rmi local --remove-orphans
