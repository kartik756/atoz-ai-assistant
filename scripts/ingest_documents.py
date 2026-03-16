"""
Document Ingestion CLI — Phase 2 Custom RAG

Entry point for the full document ingestion pipeline.

Usage:
    # Ingest all documents from S3 bucket
    python scripts/ingest_documents.py

    # Ingest documents under a specific S3 prefix
    python scripts/ingest_documents.py --prefix hr-policies/

    # Dry run — shows what would be ingested without storing anything
    python scripts/ingest_documents.py --dry-run

    # Custom chunk size
    python scripts/ingest_documents.py --chunk-size 800 --chunk-overlap 150

Pipeline:
    S3 bucket
        → DocumentLoader    (download + parse PDF/TXT)
        → DocumentChunker   (split into overlapping chunks)
        → EmbeddingPipeline (embed via Titan + store in OpenSearch)

Run this script:
    - Once before first use (populates OpenSearch index)
    - Whenever new documents are uploaded to S3
    - Whenever existing documents are updated (chunks are overwritten via
      deterministic IDs — no duplicates created)
"""

import asyncio
import argparse
import logging
import sys
import os
import time
from typing import Optional

# Add project root to path so imports work from scripts/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunking import DocumentChunker
from rag.ingestion.embeddings import EmbeddingPipeline

# ── Logging setup ──────────────────────────────────────────────────────────
# Scripts use a simpler format than the API (no request IDs needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# ───────────────────────────────────────────────────────────────────────────


async def run_ingestion(
    prefix: str = "",
    dry_run: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> None:
    """
    Execute the full ingestion pipeline end to end.

    Args:
        prefix:       S3 key prefix to filter documents e.g. "hr-policies/"
        dry_run:      If True, run loader + chunker but skip embedding + storage
        chunk_size:   Characters per chunk (default 500)
        chunk_overlap: Overlap between chunks (default 100)
    """
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("AtoZ AI Assistant — Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"S3 bucket    : {settings.S3_BUCKET_NAME}")
    logger.info(f"S3 prefix    : '{prefix}' (empty = all documents)")
    logger.info(f"OpenSearch   : {settings.opensearch_endpoint}")
    logger.info(f"Index        : {settings.OPENSEARCH_INDEX}")
    logger.info(f"Chunk size   : {chunk_size} chars")
    logger.info(f"Chunk overlap: {chunk_overlap} chars")
    logger.info(f"Dry run      : {dry_run}")
    logger.info("=" * 60)

    start_time = time.time()

    # ── Stage 1: Load documents from S3 ───────────────────────────────────
    logger.info("\n[Stage 1/3] Loading documents from S3...")

    loader = DocumentLoader()
    documents = loader.load_from_s3(prefix=prefix)

    if not documents:
        logger.warning(
            f"No supported documents found in "
            f"s3://{settings.S3_BUCKET_NAME}/{prefix}"
        )
        logger.warning("Upload PDF or TXT files to S3 and retry.")
        return

    logger.info(f"Loaded {len(documents)} documents:")
    for doc in documents:
        logger.info(f"  - {doc['filename']} ({doc['s3_key']})")

    # ── Stage 2: Chunk documents ───────────────────────────────────────────
    logger.info("\n[Stage 2/3] Chunking documents...")

    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = chunker.chunk_documents(documents)

    logger.info(f"Produced {len(chunks)} total chunks")

    # Show per-document chunk breakdown
    chunk_counts: dict = {}
    for chunk in chunks:
        fname = chunk["metadata"]["source_file"]
        chunk_counts[fname] = chunk_counts.get(fname, 0) + 1

    for filename, count in chunk_counts.items():
        logger.info(f"  - {filename}: {count} chunks")

    # ── Dry run exits here ─────────────────────────────────────────────────
    if dry_run:
        logger.info(
            "\nDry run complete. "
            "No data written to OpenSearch. "
            "Remove --dry-run to run full ingestion."
        )
        return

    # ── Stage 3: Embed and store in OpenSearch ─────────────────────────────
    logger.info("\n[Stage 3/3] Embedding chunks and storing in OpenSearch...")
    logger.info(
        f"This may take a few minutes for large document sets. "
        f"({len(chunks)} chunks × Bedrock API calls)"
    )

    embedding_pipeline = EmbeddingPipeline()
    result = await embedding_pipeline.embed_and_store(chunks)

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Complete")
    logger.info("=" * 60)
    logger.info(f"Documents processed : {len(documents)}")
    logger.info(f"Chunks produced     : {len(chunks)}")
    logger.info(f"Successfully stored : {result['success']}")
    logger.info(f"Failed              : {result['failed']}")
    logger.info(f"Time elapsed        : {elapsed}s")
    logger.info("=" * 60)

    # Exit with error code if any chunks failed
    # Allows CI/CD pipelines to detect partial failures
    if result["failed"] > 0:
        logger.error(
            f"{result['failed']} chunks failed to ingest. "
            "Check logs above for details."
        )
        sys.exit(1)

    logger.info("OpenSearch index is ready. API can now serve queries.")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Examples:
        python scripts/ingest_documents.py
        python scripts/ingest_documents.py --prefix hr-policies/
        python scripts/ingest_documents.py --dry-run
        python scripts/ingest_documents.py --chunk-size 800 --chunk-overlap 150
    """
    parser = argparse.ArgumentParser(
        description="AtoZ AI Assistant — Document Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_documents.py
  python scripts/ingest_documents.py --prefix hr-policies/
  python scripts/ingest_documents.py --dry-run
  python scripts/ingest_documents.py --chunk-size 800 --chunk-overlap 150
        """
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 key prefix to filter documents (default: all documents)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run loader and chunker only — skip embedding and storage"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Characters per chunk (default: 500)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap characters between chunks (default: 100)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        run_ingestion(
            prefix=args.prefix,
            dry_run=args.dry_run,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    )