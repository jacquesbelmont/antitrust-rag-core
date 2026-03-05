from __future__ import annotations
import uuid


class Uuid4IdGenerator:
    def new_id(self) -> str:
        return uuid.uuid4().hex
