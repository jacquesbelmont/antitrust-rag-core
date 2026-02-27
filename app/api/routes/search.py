from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.application.dto import SearchQuery, SearchResult
from app.application.rag_service import RAGService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResult)
async def search(
    q: SearchQuery,
    rag: RAGService = Depends(get_rag_service),
) -> SearchResult:
    answer, context_items, evaluation = await rag.answer(query=q.query, top_k=q.top_k)
    return SearchResult(answer=answer, context=context_items, evaluation=evaluation)
