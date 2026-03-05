-- ──────────────────────────────────────────────────────────────────────────────
-- infra/postgres/01_init.sql
-- Runs once on first container start (docker-entrypoint-initdb.d/)
--
-- Purpose: apply Principle of Least Privilege — the app role has NO superuser
-- rights. DDL (CREATE TABLE) is handled by Alembic migrations, not the app.
-- ──────────────────────────────────────────────────────────────────────────────

-- The POSTGRES_USER defined in compose already owns the database.
-- We create a separate, more restricted app role for runtime connections.

-- 1. App role — only DML rights (SELECT, INSERT, UPDATE, DELETE)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'legalrag_app') THEN
        CREATE ROLE legalrag_app WITH LOGIN PASSWORD 'PLACEHOLDER_replaced_by_alembic_migration';
    END IF;
END
$$;

-- 2. Grant connect on the database
GRANT CONNECT ON DATABASE legalrag TO legalrag_app;

-- 3. Grant usage on the public schema
GRANT USAGE ON SCHEMA public TO legalrag_app;

-- 4. Default privileges: any table created by the owner is accessible to the app role
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO legalrag_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO legalrag_app;

-- NOTE: The legalrag (owner) role handles CREATE TABLE via Alembic.
--       The legalrag_app role is what the API and Worker use at runtime.
--       Separate DATABASE_URL env vars should be set per service.
