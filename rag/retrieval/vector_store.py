from services.vector_service import VectorService


class VectorStore:
    """
    Abstraction layer over the vector database.

    This class interacts with VectorService which
    communicates with OpenSearch.
    """

    def __init__(self):
        self.vector_service = VectorService()

    def add_documents(self, documents: list):
        """
        Store document embeddings into the vector database.
        """

        self.vector_service.index_documents(documents)

    def search(self, query_embedding: list, k: int = 5):
        """
        Perform vector similarity search.
        """

        return self.vector_service.similarity_search(query_embedding, k)