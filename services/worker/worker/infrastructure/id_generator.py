from __future__ import annotations
import uuid


class Uuid4IdGenerator:
    """Generates random UUID4 strings (without hyphens for compact storage)."""

    def new_id(self) -> str:
        return uuid.uuid4().hex
