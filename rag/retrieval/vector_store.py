"""
Vector Store — Retrieval-Side OpenSearch Wrapper. In simple, VectorStore = search interface for vector database.

Responsibilities:
- Provide a clean retrieval interface over VectorService
- Separate retrieval concerns from ingestion concerns
- Single place to add retrieval-specific logic (filters, hybrid search later)

Why separate from services/vector_service.py:
- vector_service.py owns index management + ingestion (write path)
- vector_store.py owns retrieval (read path)
- Clean separation of read vs write responsibilities
- Swap underlying store later without touching retriever.py
"""

import logging
from typing import List, Dict, Any

from services.vector_service import VectorService

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Read-side interface to the OpenSearch vector index.

    Used exclusively by the retrieval layer.
    Ingestion uses VectorService directly.
    """

    def __init__(self):
        self.vector_service = VectorService()

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar chunks to the query vector.

        Called by: rag/retrieval/retriever.py

        Args:
            query_vector: Embedded user query (1536-dim float list)
            top_k:        Number of chunks to return

        Returns:
            List of dicts:
            {
                "text":     str,   chunk text — used as LLM context
                "score":    float, cosine similarity (0–1)
                "metadata": dict   source file, chunk index, s3 key
            }
        """
        logger.info(f"Running similarity search, top_k={top_k}")

        results = self.vector_service.search(
            query_vector=query_vector,
            top_k=top_k
        )

        logger.info(f"Similarity search returned {len(results)} results")
        return results

    def health_check(self) -> bool:
        """Proxy health check to underlying vector service."""
        return self.vector_service.health_check()