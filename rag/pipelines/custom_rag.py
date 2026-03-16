"""
Custom RAG Pipeline — Phase 2 Orchestrator

Responsibilities:
- Coordinate the full custom RAG flow end to end
- Retrieve relevant chunks via Retriever
- Build a structured prompt with retrieved context
- Generate answer via BedrockService
- Return answer with source citations

This is the top-level pipeline for custom RAG.
The API route calls this. Nothing below this layer
knows about HTTP requests or API schemas.

Flow:
    run(query)
        → Retriever.retrieve(query)
            → embed query
            → similarity search OpenSearch
            → filter by score
        → _build_prompt(query, chunks)
        → BedrockService.generate_response(query, context)
        → return CustomRAGResponse
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from services.bedrock_service import BedrockService
from rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


# ── Response shape ─────────────────────────────────────────────────────────

@dataclass  #automatically creates classes for storing data
class CustomRAGResponse:
    """
    Structured response returned by the custom RAG pipeline.

    answer:  Generated answer from the LLM
    sources: List of source chunks used as context
             Each source includes text, score, and metadata
             so the API can return citations to the user
    """
    answer: str
    sources: List[Dict[str, Any]]

# ───────────────────────────────────────────────────────────────────────────


class CustomRAGPipeline:
    """
    Orchestrates the full custom RAG pipeline.

    Instantiated once at app startup via FastAPI dependency injection.
    Shared across all requests — stateless by design.
    """

    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)
        self.bedrock_service = BedrockService()

    async def run(self, query: str) -> CustomRAGResponse:
        """
        Execute the full custom RAG pipeline for a user query.

        Called by: api/routes/chat.py when pipeline = "custom"

        Args:
            query: Raw user question from the API request

        Returns:
            CustomRAGResponse with answer and source citations

        Raises:
            ValueError: If query is empty
            Exception:  Propagates retrieval or generation failures
                        so the API layer can handle them appropriately
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Custom RAG pipeline started | query: '{query}'")

        # ── Step 1: Retrieve relevant chunks ──────────────────────────────
        chunks = await self.retriever.retrieve(query)

        # Handle case where no relevant chunks found
        # Don't call LLM with empty context — return honest response
        if not chunks:
            logger.warning(
                f"No relevant chunks found for query: '{query}'. "
                "Returning fallback response."
            )
            return CustomRAGResponse(
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "to answer your question. Please try rephrasing or contact "
                    "HR directly for assistance."
                ),
                sources=[]
            )

        logger.info(f"Retrieved {len(chunks)} chunks for context")

        # ── Step 2: Extract text for LLM context ──────────────────────────
        # Pass plain text list to generate_response (matches Phase 1 signature)
        context_texts = [chunk["text"] for chunk in chunks]

        # ── Step 3: Generate answer via Bedrock LLM ───────────────────────
        answer = await self.bedrock_service.generate_response(
            query=query,
            context=context_texts
        )

        # ── Step 4: Build structured source citations ──────────────────────
        sources = self._build_sources(chunks)

        logger.info("Custom RAG pipeline complete")

        return CustomRAGResponse(
            answer=answer,
            sources=sources
        )

    def _build_sources(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build clean source citation objects from retrieved chunks.

        These are returned to the API and ultimately to the user
        so they can verify which documents the answer came from.

        Real enterprise assistants always show sources — it builds
        trust and lets employees verify policy information themselves.

        Args:
            chunks: Raw chunks from Retriever (text, score, metadata)

        Returns:
            List of clean source dicts:
            {
                "text":        str,   chunk text shown as citation
                "score":       float, rounded similarity score
                "source_file": str,   original filename
                "source":      str,   full S3 URI
                "chunk_index": int    position in original document
            }
        """
        sources = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            sources.append({
                "text":        chunk["text"],
                "score":       round(chunk["score"], 4),
                "source_file": metadata.get("source_file", "unknown"),
                "source":      metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1)
            })

        return sources