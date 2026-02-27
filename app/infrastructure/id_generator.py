from __future__ import annotations

import uuid


class Uuid7LikeIdGenerator:
    def new_id(self) -> str:
        return uuid.uuid4().hex
