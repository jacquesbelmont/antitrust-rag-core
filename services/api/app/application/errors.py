"""
Application-layer exceptions.

These are raised by service/application code and caught by FastAPI exception
handlers in ``main.py``, which map them to appropriate HTTP status codes.
Stack traces are never forwarded to the client.
"""
from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""


class ValidationError(AppError):
    """Input failed domain-level validation (distinct from Pydantic's)."""


class NotFoundError(AppError):
    """A required resource was not found."""


class StorageError(AppError):
    """A storage backend (PG, Weaviate) operation failed."""


class PromptInjectionError(ValidationError):
    """The query contains patterns that indicate a prompt injection attempt."""
