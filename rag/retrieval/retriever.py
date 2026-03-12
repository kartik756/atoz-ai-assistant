from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.vector_store import VectorStore


class Retriever:
    """
    Retriever responsible for fetching relevant
    document chunks for a user query.
    """

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve top-k relevant documents for the query.
        """

        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embeddings(query)

        # Search vector database
        results = self.vector_store.search(query_embedding, k)

        return results