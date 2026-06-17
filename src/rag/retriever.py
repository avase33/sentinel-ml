"""
VectorRetriever — ChromaDB-backed document retrieval with embedding caching.
"""

import hashlib
import logging
from typing import Any

from chromadb import AsyncClientAPI
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Simple in-process LRU cache for query embeddings (avoids re-encoding repeated queries)
_EMBED_CACHE: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 1_000


class VectorRetriever:
    """
    Retrieves semantically similar documents from ChromaDB.

    Embedding caching: query embeddings are cached in memory by query hash.
    For high-traffic use, replace with Redis-backed cache.
    """

    def __init__(self, chroma_client: AsyncClientAPI, embedder: SentenceTransformer):
        self._chroma = chroma_client
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[Document]:
        """
        Embed the query and return the top_k most similar documents.
        Optionally filter by metadata using ChromaDB `where` syntax.
        """
        query_embedding = self._embed(query)
        col = await self._chroma.get_or_create_collection(collection)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = await col.query(**kwargs)

        documents = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance   # cosine distance → similarity
            documents.append(
                Document(
                    page_content=doc_text,
                    metadata={**metadata, "score": round(score, 4)},
                )
            )

        return documents

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def upsert(
        self,
        documents: list[Document],
        collection: str,
        batch_size: int = 100,
    ) -> int:
        """
        Embed and store documents. Uses batched embedding for efficiency.
        Documents are upserted by content hash (idempotent).
        """
        col = await self._chroma.get_or_create_collection(collection)
        total = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [d.page_content for d in batch]
            metas = [d.metadata for d in batch]
            ids   = [self._doc_id(t) for t in texts]

            embeddings = self._embedder.encode(texts, normalize_embeddings=True).tolist()

            await col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metas,
            )
            total += len(batch)
            logger.debug("Upserted batch %d/%d (%d docs)", i // batch_size + 1, -(-len(documents) // batch_size), len(batch))

        logger.info("Upserted %d documents into collection '%s'", total, collection)
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Embed a single query string with in-process caching."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in _EMBED_CACHE:
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                # Evict oldest entry (simple FIFO)
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
            embedding = self._embedder.encode([text], normalize_embeddings=True)[0]
            _EMBED_CACHE[key] = embedding.tolist()
        return _EMBED_CACHE[key]

    @staticmethod
    def _doc_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]
