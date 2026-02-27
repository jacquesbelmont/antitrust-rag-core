import pytest

from app.application.context_formatting import ContextFormatter
from app.application.evaluation import EvaluationService
from app.application.rag_service import RAGService
from app.application.retrieval_service import RetrievalService
from app.domain.entities import Chunk
from app.infrastructure.embedding import DeterministicHashEmbedder
from app.infrastructure.llm_mock import MockLLMClient
from app.infrastructure.vector_store_in_memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_rag_returns_answer_and_metrics():
    store = InMemoryVectorStore()
    embedder = DeterministicHashEmbedder()

    c1 = Chunk(id="c1", document_id="d1", text="antitrust law concerns competition", index=0, hierarchy_path=["TITLE I"], metadata={})
    c1_vec = await embedder.embed_text(text=c1.text)
    c1 = Chunk(**{**c1.__dict__, "metadata": {"vector": c1_vec}})

    await store.upsert_chunks(chunks=[c1])

    retrieval = RetrievalService(vector_store=store, embedder=embedder)
    rag = RAGService(
        retrieval=retrieval,
        formatter=ContextFormatter(),
        llm=MockLLMClient(),
        evaluation=EvaluationService(),
    )

    answer, context_items, evaluation = await rag.answer(query="What is antitrust?", top_k=1)

    assert "MOCK_ANSWER" in answer
    assert len(context_items) == 1
    assert "context_relevance" in evaluation
    assert "faithfulness_proxy" in evaluation
