"""
Retriever — Query-Time Orchestration

Responsibilities:
- Accept a raw text query from the RAG pipeline
- Embed the query using BedrockService
- Search OpenSearch via VectorStore
- Return clean context chunks ready for LLM consumption

This is the entry point for ALL retrieval in the custom RAG pipeline.
The RAG pipeline calls retrieve() and gets back text chunks — it never
touches embeddings or OpenSearch directly.

When it comes into picture:
    User sends question
        → custom_rag.py calls retriever.retrieve(query)
            → embed query via Bedrock Titan
            → similarity search in OpenSearch
            → return top-k chunks as plain text + metadata
        → custom_rag.py passes chunks to LLM as context
"""

import logging
from typing import List, Dict, Any

from services.bedrock_service import BedrockService
from rag.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Orchestrates the full query-time retrieval flow.

    Flow:
        raw query (str)
            → BedrockService.embed_text()    → query vector
            → VectorStore.similarity_search() → top-k chunks
            → return chunks to RAG pipeline
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.bedrock_service = BedrockService()
        self.vector_store = VectorStore()

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Full retrieval flow: text query → embedded → searched → chunks returned.

        Called by: rag/pipelines/custom_rag.py

        Args:
            query: Raw user question e.g. "What is the leave policy?"

        Returns:
            List of chunk dicts ranked by similarity:
            {
                "text":     str,   chunk text passed to LLM as context
                "score":    float, relevance score (higher = more relevant)
                "metadata": dict   source_file, chunk_index, s3_key, source
            }
        """
        logger.info(f"Retrieval started for query: '{query}'")

        # Step 1: Embed the user's question into a vector
        query_vector = await self.bedrock_service.embed_text(query)
        logger.info("Query embedded successfully")

        # Step 2: Find most similar chunks in OpenSearch
        results = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=self.top_k
        )

        # Step 3: Filter out low-confidence results
        # Score threshold prevents hallucination from irrelevant chunks
        # 0.3 is conservative — tune based on your document quality
        filtered = self._filter_by_score(results, min_score=0.3)

        logger.info(
            f"Retrieval complete — {len(results)} raw results, "
            f"{len(filtered)} after score filtering"
        )

        return filtered

    def _filter_by_score(
        self,
        results: List[Dict[str, Any]],
        min_score: float
    ) -> List[Dict[str, Any]]:
        """
        Remove chunks below the minimum similarity threshold.

        Why this matters:
        If a user asks something completely unrelated to the knowledge base,
        OpenSearch still returns top_k results — they just have very low scores.
        Without this filter, those irrelevant chunks would be fed to the LLM
        as context, causing hallucinated or misleading answers.

        Args:
            results:   Raw results from VectorStore
            min_score: Minimum cosine similarity score (0.0 – 1.0)

        Returns:
            Filtered list containing only high-confidence chunks
        """
        filtered = [r for r in results if r["score"] >= min_score]

        if not filtered:
            logger.warning(
                f"All results scored below {min_score}. "
                "Query may be outside the knowledge base scope."
            )

        return filtered