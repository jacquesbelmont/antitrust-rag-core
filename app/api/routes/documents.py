from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_ingestion_service
from app.application.dto import IngestDocumentCommand, IngestDocumentResult
from app.application.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=IngestDocumentResult)
async def ingest_document(
    cmd: IngestDocumentCommand,
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> IngestDocumentResult:
    document_id, chunks_created = await ingestion.ingest(title=cmd.title, source=cmd.source, text=cmd.text)
    return IngestDocumentResult(document_id=document_id, chunks_created=chunks_created)
