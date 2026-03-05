"""Initial schema — documents, chunks, ingestion_jobs

Revision ID: 0001
Revises:
Create Date: 2026-03-02

This is the baseline migration.  It creates:
  - PostgreSQL ENUM types: document_status, ingestion_job_status
  - Tables:  documents, chunks, ingestion_jobs
  - Indexes: ix_chunks_document_id, ix_ingestion_jobs_document_id,
             ix_ingestion_jobs_status

Run with:
    ALEMBIC_DATABASE_URL=postgresql+psycopg2://... \\
        alembic -c infra/migrations/alembic.ini upgrade head
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# ── Revision identifiers ──────────────────────────────────────────────────────
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


# ── Enum types (created before tables that reference them) ────────────────────
document_status_enum = postgresql.ENUM(
    "pending", "processing", "completed", "failed",
    name="document_status",
    create_type=False,   # we create manually below for explicit control
)
ingestion_job_status_enum = postgresql.ENUM(
    "pending", "started", "success", "failure", "retry", "revoked",
    name="ingestion_job_status",
    create_type=False,
)


def upgrade() -> None:
    # ── 1. Create ENUM types ──────────────────────────────────────────────────
    document_status_enum.create(op.get_bind(), checkfirst=True)
    ingestion_job_status_enum.create(op.get_bind(), checkfirst=True)

    # ── 2. documents ──────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id",          sa.String(36),  primary_key=True),
        sa.Column("title",       sa.Text(),       nullable=True),
        sa.Column("source",      sa.Text(),       nullable=True),
        # Full document text — PostgreSQL will TOAST-compress automatically.
        sa.Column("raw_text",    sa.Text(),       nullable=False),
        sa.Column("char_count",  sa.Integer(),    nullable=False, server_default="0"),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending", "processing", "completed", "failed",
                name="document_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # ── 3. chunks ─────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id",           sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text",          sa.Text(),     nullable=False),
        sa.Column("chunk_index",   sa.Integer(),  nullable=False),
        # JSONB for indexed, structured storage of hierarchy and metadata.
        sa.Column(
            "hierarchy_path",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # Populated by the worker after successful Weaviate upsert.
        sa.Column("vector_id",    sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])

    # ── 4. ingestion_jobs ─────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        # id IS the Celery task UUID — no extra join needed.
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending", "started", "success", "failure", "retry", "revoked",
                name="ingestion_job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("chunks_created",  sa.Integer(), nullable=True),
        sa.Column("error_message",   sa.Text(),    nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("started_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_status",      "ingestion_jobs", ["status"])

    # ── 5. updated_at trigger ─────────────────────────────────────────────────
    # PostgreSQL does not update `updated_at` automatically on UPDATE unless
    # there is a trigger.  The trigger below fires on every UPDATE to the
    # `documents` table, keeping `updated_at` accurate even for writes that
    # bypass SQLAlchemy's `onupdate` (e.g. raw psql admin edits).
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER trg_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at")

    op.drop_index("ix_ingestion_jobs_status",      table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")

    op.drop_table("documents")

    ingestion_job_status_enum.drop(op.get_bind(), checkfirst=True)
    document_status_enum.drop(op.get_bind(), checkfirst=True)
