"""
/api/v1/rag  — LLM / RAG gateway endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag.pipeline import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the refund policy for recurring payments?")
    collection: str = Field("default", example="compliance_docs")
    top_k: int = Field(5, ge=1, le=20)
    provider: str = Field("openai", pattern="^(openai|anthropic)$")
    system_prompt: str | None = None
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    texts: list[str]
    metadatas: list[dict] | None = None
    collection: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    provider: str
    model: str
    latency_ms: float
    tokens_used: int | None = None


@router.post("/rag/query", response_model=QueryResponse)
async def rag_query(payload: QueryRequest):
    """
    RAG-powered question answering.
    Embeds the query, retrieves top_k documents, and generates a grounded answer.
    """
    try:
        result = await pipeline.query(
            query=payload.query,
            collection=payload.collection,
            top_k=payload.top_k,
            provider=payload.provider,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
    )


@router.post("/rag/ingest", summary="Ingest documents into the vector store")
async def ingest_documents(payload: IngestRequest):
    """
    Embed and store text documents in ChromaDB.
    Documents are chunked by the caller — use LangChain splitters for large files.
    """
    try:
        n = await pipeline.ingest_texts(
            texts=payload.texts,
            metadatas=payload.metadatas,
            collection=payload.collection,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ok", "documents_ingested": n, "collection": payload.collection}
