"""
RAG Pipeline — multi-provider LLM gateway with vector-backed retrieval.

Supports:
  - OpenAI (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
  - Anthropic (claude-3-5-sonnet, claude-3-haiku)
  - Local models via Ollama (optional)

Vector store backend: ChromaDB (local) or Pinecone (cloud) — configured via env.
"""

import logging
import os
import time
from dataclasses import dataclass

import anthropic
from chromadb import AsyncHttpClient as ChromaAsyncClient
from langchain.schema import Document
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.rag.retriever import VectorRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CHROMA_HOST       = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
EMBED_MODEL       = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # fast, 384-dim

PROVIDER_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]   # [{content, metadata, score}]
    provider: str
    model: str
    latency_ms: float
    tokens_used: int | None = None


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG: embed query → retrieve top-k docs → build prompt → generate.

    Lazy-initializes the embedding model and vector store client on first use
    to keep startup time fast.
    """

    def __init__(self):
        self._embedder: SentenceTransformer | None = None
        self._retriever: VectorRetriever | None = None
        self._openai = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self._anthropic = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

    async def _get_retriever(self) -> VectorRetriever:
        if self._retriever is None:
            chroma = await ChromaAsyncClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self._embedder = SentenceTransformer(EMBED_MODEL)
            self._retriever = VectorRetriever(chroma_client=chroma, embedder=self._embedder)
        return self._retriever

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    async def query(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        provider: str = "openai",
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> RAGResponse:
        """
        Full RAG pipeline:
          1. Embed the query.
          2. Retrieve top_k documents from the vector store.
          3. Build a grounded prompt with retrieved context.
          4. Generate an answer from the specified LLM provider.
        """
        start = time.perf_counter()

        retriever = await self._get_retriever()
        docs = await retriever.retrieve(
            query=query,
            collection=collection,
            top_k=top_k,
        )

        context = self._build_context(docs)
        system = system_prompt or self._default_system_prompt()
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        if provider == "openai":
            answer, tokens = await self._call_openai(system, user_message, temperature)
        elif provider == "anthropic":
            answer, tokens = await self._call_anthropic(system, user_message, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}. Choose 'openai' or 'anthropic'.")

        latency_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=answer,
            sources=[{"content": d.page_content, "metadata": d.metadata, "score": d.metadata.get("score")} for d in docs],
            provider=provider,
            model=PROVIDER_MODELS[provider],
            latency_ms=round(latency_ms, 2),
            tokens_used=tokens,
        )

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self,
        documents: list[Document],
        collection: str,
        batch_size: int = 100,
    ) -> int:
        """
        Embed and store documents in the vector store.
        Returns the number of documents ingested.
        """
        retriever = await self._get_retriever()
        return await retriever.upsert(documents=documents, collection=collection, batch_size=batch_size)

    async def ingest_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        collection: str = "default",
    ) -> int:
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        return await self.ingest(docs, collection)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _call_openai(self, system: str, user_msg: str, temperature: float) -> tuple[str, int]:
        if not self._openai:
            raise RuntimeError("OPENAI_API_KEY not set")
        resp = await self._openai.chat.completions.create(
            model=PROVIDER_MODELS["openai"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    async def _call_anthropic(self, system: str, user_msg: str, temperature: float) -> tuple[str, int]:
        if not self._anthropic:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        msg = await self._anthropic.messages.create(
            model=PROVIDER_MODELS["anthropic"],
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        return msg.content[0].text, tokens

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        if not docs:
            return "No relevant context found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a precise, factual assistant. Answer the user's question "
            "using ONLY the provided context. If the context does not contain "
            "enough information to answer, say so clearly. Do not hallucinate. "
            "Cite the source numbers [1], [2], etc. when referencing context."
        )
