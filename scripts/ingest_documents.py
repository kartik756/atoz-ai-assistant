from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunking import TextChunker
from rag.ingestion.embeddings import EmbeddingGenerator
from rag.retrieval.vector_store import VectorStore
from services.vector_service import VectorService


def ingest_documents(folder_path: str):
    """
    Run document ingestion pipeline.
    """

    # Step 1: Create vector index if not exists
    vector_service = VectorService()
    vector_service.create_index()

    # Step 2: Load documents
    loader = DocumentLoader()
    documents = loader.load_documents(folder_path)

    # Step 3: Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)

    # Step 4: Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embedded_docs = embedding_generator.generate_embeddings(chunks)

    # Step 5: Store in vector database
    vector_store = VectorStore()
    vector_store.add_documents(embedded_docs)

    print("Document ingestion completed successfully")


if __name__ == "__main__":
    ingest_documents("data/documents")