"""
Document Chunker — Text Splitting Pipeline

Responsibilities:
- Split large documents into smaller, semantically focused chunks
- Maintain overlap between chunks to preserve context at boundaries
- Attach metadata to each chunk for traceability back to source document

Why chunking matters:
- LLMs have token limits — can't process entire documents
- Smaller chunks produce better, more focused vector embeddings
- Overlap prevents context loss at chunk boundaries
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Chunking configuration ─────────────────────────────────────────────────
# These values are tuned for HR policy documents.
# Adjust per document type if needed in future phases.

CHUNK_SIZE = 500        # characters per chunk
                        # ~350-400 tokens for English prose
                        # fits comfortably in Titan's 8192 token input limit

CHUNK_OVERLAP = 100     # characters shared between adjacent chunks
                        # prevents cutting a sentence or policy clause mid-way

# ───────────────────────────────────────────────────────────────────────────


class DocumentChunker:
    """
    Splits raw document text into overlapping chunks using
    LangChain's RecursiveCharacterTextSplitter.

    Splitting priority (tries each separator in order):
        1. Paragraph breaks  (\n\n)
        2. Line breaks       (\n)
        3. Sentences         (. )
        4. Words             ( )
        5. Characters        (last resort)

    This hierarchy ensures chunks break at natural language boundaries
    rather than mid-word or mid-sentence.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,        # character-based length
                                        # consistent across all text types
        )

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of loaded documents into smaller pieces.

        Called by: rag/ingestion/embeddings.py (or directly from ingest script)

        Args:
            documents: Output from DocumentLoader.load_from_s3()
                       Each doc must have 'text', 'filename', 'source', 's3_key'

        Returns:
            List of chunk dicts:
            {
                "id":       str,   deterministic chunk ID  e.g. "leave_policy.pdf_chunk_0"
                "text":     str,   chunk text content
                "metadata": dict   source traceability info
            }
        """
        all_chunks = []

        for doc in documents:
            chunks = self._chunk_single_document(doc)
            all_chunks.extend(chunks)
            logger.info(
                f"{doc['filename']} → {len(chunks)} chunks"
            )

        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    def _chunk_single_document(
        self,
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks with metadata attached.

        Args:
            document: Single document dict from DocumentLoader

        Returns:
            List of chunk dicts for this document
        """
        raw_text = document["text"]
        filename = document["filename"]

        # Split text into raw string chunks
        text_chunks = self.splitter.split_text(raw_text)

        chunks = []

        for index, chunk_text in enumerate(text_chunks):

            chunk = {
                # Deterministic ID: same document always produces same IDs
                # Safe to re-ingest — overwrites instead of duplicating
                "id": f"{filename}_chunk_{index}",

                "text": chunk_text.strip(),

                "metadata": {
                    "source_file": filename,
                    "s3_key":      document["s3_key"],
                    "source":      document["source"],
                    "chunk_index": index,
                    "total_chunks": len(text_chunks)
                }
            }

            chunks.append(chunk)

        return chunks