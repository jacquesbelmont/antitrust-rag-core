from __future__ import annotations
from datetime import datetime, timezone


class UtcClock:
    """Clock that returns the current UTC datetime."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)
