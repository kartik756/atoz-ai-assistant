"""
Embedding Pipeline — Chunk Vectorization and Storage

Responsibilities:
- Receive chunks from the chunking pipeline
- Generate vector embeddings for each chunk via BedrockService
- Batch chunks to respect Bedrock API rate limits
- Store embedded chunks in OpenSearch via VectorService

This is the bridge between the ingestion pipeline and the vector store.
"""

import asyncio
import logging
from typing import List, Dict, Any

from services.bedrock_service import BedrockService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)

# Bedrock Titan Embeddings rate limit is 2000 RPM on most accounts.
# Batch size of 20 with 0.5s delay = ~40 req/s = safe headroom.
# Increase batch size after confirming your account limits.
BATCH_SIZE = 20
BATCH_DELAY_SECONDS = 0.5


class EmbeddingPipeline:
    """
    Orchestrates embedding generation and storage for document chunks.

    Takes output of DocumentChunker and produces populated OpenSearch index.
    """

    def __init__(self):
        self.bedrock_service = BedrockService()
        self.vector_service = VectorService()

    async def embed_and_store(
        self,
        chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Embed all chunks and store them in OpenSearch.

        Called by: scripts/ingest_documents.py

        Args:
            chunks: Output from DocumentChunker.chunk_documents()
                    Each chunk must have 'id', 'text', 'metadata'

        Returns:
            { "total": int, "success": int, "failed": int }
        """
        if not chunks:
            logger.warning("embed_and_store called with empty chunk list")
            return {"total": 0, "success": 0, "failed": 0}

        logger.info(f"Starting embedding pipeline for {len(chunks)} chunks")

        # Ensure index exists before inserting anything
        self.vector_service.ensure_index_exists()

        total_success = 0
        total_failed = 0

        # Process in batches to respect rate limits
        batches = self._create_batches(chunks, BATCH_SIZE)

        for batch_num, batch in enumerate(batches, start=1):

            logger.info(
                f"Processing batch {batch_num}/{len(batches)} "
                f"({len(batch)} chunks)"
            )

            embedded_batch = await self._embed_batch(batch)

            if embedded_batch:
                result = self.vector_service.upsert_documents(embedded_batch)
                total_success += result["success"]
                total_failed += result["failed"]

            # Respect rate limits between batches
            if batch_num < len(batches):
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        summary = {
            "total": len(chunks),
            "success": total_success,
            "failed": total_failed
        }

        logger.info(f"Embedding pipeline complete: {summary}")
        return summary

    async def _embed_batch(
        self,
        batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a single batch of chunks concurrently.

        Runs all embed_text() calls in the batch simultaneously
        using asyncio.gather() — much faster than sequential calls.

        Args:
            batch: List of chunk dicts

        Returns:
            Batch with 'embedding' field added to each chunk.
            Chunks that fail to embed are excluded (logged, not raised).
        """
        # Fire all embedding calls concurrently within the batch
        tasks = [
            self.bedrock_service.embed_text(chunk["text"])
            for chunk in batch
        ]

        # return_exceptions=True — failed embeds return Exception objects
        # instead of crashing the whole batch
        results = await asyncio.gather(*tasks, return_exceptions=True)

        embedded = []

        for chunk, result in zip(batch, results):

            if isinstance(result, Exception):
                logger.error(
                    f"Embedding failed for chunk '{chunk['id']}': {result}"
                )
                continue    # skip this chunk, don't crash the batch

            embedded.append({
                "id":        chunk["id"],
                "text":      chunk["text"],
                "embedding": result,        # List[float] from Titan
                "metadata":  chunk["metadata"]
            })

        return embedded

    def _create_batches(
        self,
        items: List[Any],
        batch_size: int
    ) -> List[List[Any]]:
        """
        Split a flat list into batches of given size.
        Standard utility — used in every production ingestion pipeline.
        """
        return [
            items[i: i + batch_size]
            for i in range(0, len(items), batch_size)
        ]