from typing import List, Dict


class TextChunker:
    """
    Responsible for splitting documents into chunks
    suitable for embedding and vector storage.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of raw documents

        Returns:
            List of chunked documents
        """

        chunks = []

        for doc in documents:

            text = doc["text"]
            metadata = doc["metadata"]

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk_text = text[start:end]

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": metadata
                    }
                )

                start += self.chunk_size - self.overlap

        return chunks