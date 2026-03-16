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

Note on implementation:
- Pure Python implementation, no LangChain dependency
- Separator hierarchy mirrors RecursiveCharacterTextSplitter logic
- Owned internally for full control and zero external risk
"""

import logging
import re
from typing import List, Dict, Any

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Chunking configuration ─────────────────────────────────────────────────
CHUNK_SIZE = 500        # characters per chunk (~350-400 tokens for English prose)
CHUNK_OVERLAP = 100     # characters shared between adjacent chunks
# ───────────────────────────────────────────────────────────────────────────

# Separator hierarchy — tried in order, falls through to next if chunk still too large
# This mirrors RecursiveCharacterTextSplitter's core logic
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class DocumentChunker:
    """
    Splits raw document text into overlapping chunks.

    Splitting priority (tries each separator in order):
        1. Paragraph breaks  (\\n\\n)
        2. Line breaks       (\\n)
        3. Sentences         (. )
        4. Words             ( )
        5. Characters        (last resort)

    Pure Python — no external dependencies.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a list of loaded documents into smaller pieces.

        Called by: rag/ingestion/embeddings.py

        Args:
            documents: Output from DocumentLoader.load_from_s3()

        Returns:
            List of chunk dicts:
            {
                "id":       str,   e.g. "leave_policy.pdf_chunk_0"
                "text":     str,   chunk text content
                "metadata": dict   source traceability info
            }
        """
        all_chunks = []

        for doc in documents:
            chunks = self._chunk_single_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"{doc['filename']} → {len(chunks)} chunks")

        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    def _chunk_single_document(
        self,
        document: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks with metadata attached.
        """
        raw_text = document["text"]
        filename = document["filename"]

        text_chunks = self._split_text(raw_text)

        chunks = []
        for index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "id": f"{filename}_chunk_{index}",
                "text": chunk_text.strip(),
                "metadata": {
                    "source_file":  filename,
                    "s3_key":       document["s3_key"],
                    "source":       document["source"],
                    "chunk_index":  index,
                    "total_chunks": len(text_chunks)
                }
            })

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively split text using separator hierarchy.

        Strategy:
        1. Try splitting by the highest-priority separator
        2. If any resulting piece is still larger than chunk_size,
           recursively split that piece with the next separator
        3. Merge small pieces back together up to chunk_size
           with overlap applied between final chunks

        Args:
            text: Raw document text

        Returns:
            List of text chunks respecting chunk_size and chunk_overlap
        """
        # Get final small splits using separator hierarchy
        splits = self._recursive_split(text, SEPARATORS)

        # Merge splits into chunks with overlap
        return self._merge_splits(splits)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Split text by trying each separator in order.
        If a piece is still too large, recurse with the next separator.
        """
        if not text:
            return []

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator == "":
            # Last resort — split into individual characters
            splits = list(text)
        else:
            splits = re.split(re.escape(separator), text)

        final_splits = []

        for split in splits:
            split = split.strip()
            if not split:
                continue

            if len(split) <= self.chunk_size:
                # Small enough — keep as-is
                final_splits.append(split)
            elif remaining_separators:
                # Still too large — recurse with next separator
                sub_splits = self._recursive_split(split, remaining_separators)
                final_splits.extend(sub_splits)
            else:
                # No more separators — force-split by character
                for i in range(0, len(split), self.chunk_size):
                    final_splits.append(split[i:i + self.chunk_size])

        return final_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into chunks of target size with overlap.

        This is the step that produces the final chunks.
        It accumulates splits until adding the next would exceed
        chunk_size, then starts a new chunk — but carries over
        the last chunk_overlap characters into the new chunk
        to preserve context across boundaries.

        Example with chunk_size=20, overlap=5:
            splits: ["Hello world", "this is", "a test", "of chunking"]
            chunk1: "Hello world this is"   (19 chars)
            chunk2: "is a test of chunking" (carries overlap from chunk1)
        """
        if not splits:
            return []

        chunks = []
        current_chunk_parts = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            # If adding this split exceeds chunk_size, finalize current chunk
            if current_length + split_length > self.chunk_size and current_chunk_parts:
                chunk_text = " ".join(current_chunk_parts)
                chunks.append(chunk_text)

                # Apply overlap: carry over tail of current chunk
                # Walk back through parts until we have chunk_overlap chars
                overlap_parts = []
                overlap_length = 0

                for part in reversed(current_chunk_parts):
                    if overlap_length >= self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_length += len(part)

                current_chunk_parts = overlap_parts
                current_length = overlap_length

            current_chunk_parts.append(split)
            current_length += split_length

        # Don't forget the last chunk
        if current_chunk_parts:
            chunks.append(" ".join(current_chunk_parts))

        return chunks