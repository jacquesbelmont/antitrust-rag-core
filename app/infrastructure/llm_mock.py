from __future__ import annotations


class MockLLMClient:
    async def generate(self, *, prompt: str) -> str:
        # In production you'd call an LLM provider here.
        # For this PoC we return a deterministic response.
        return "MOCK_ANSWER:\n" + prompt[:4000]
