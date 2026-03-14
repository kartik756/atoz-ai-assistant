"""
Vector Service — OpenSearch Integration Layer

Responsibilities:
1. Manage OpenSearch client connection (local vs AWS)
2. Create and configure the kNN vector index
3. Upsert document chunks with their embeddings
4. Perform kNN similarity search at query time

This is the ONLY file in the codebase that imports opensearchpy.
All other modules interact with OpenSearch through this service.
"""

import logging
from typing import List, Dict, Any

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorService:
    """
    Manages all interactions with OpenSearch vector database.

    Local:      connects with no auth (DISABLE_SECURITY_PLUGIN=true in Docker)
    Production: connects with AWS SigV4 request signing via IAM role
    """

    def __init__(self):
        self.index_name = settings.OPENSEARCH_INDEX
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.client = self._build_client()

    def _build_client(self) -> OpenSearch:
        """
        Build the correct OpenSearch client based on environment.

        Local:  no auth, plain HTTP
        Prod:   AWS SigV4 signed requests via IAM role
        """
        if settings.is_local:
            logger.info("Building local OpenSearch client (no auth)")
            return OpenSearch(
                hosts=[{
                    "host": settings.OPENSEARCH_HOST,
                    "port": settings.OPENSEARCH_PORT
                }],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                connection_class=RequestsHttpConnection
            )

        # Production: AWS OpenSearch Serverless with IAM auth
        # SigV4 signing is handled by the IAM role attached to the service
        # We import here so local dev doesn't need these packages
        logger.info("Building AWS OpenSearch Serverless client (SigV4 auth)")
        from opensearchpy import AWSV4SignerAuth
        import boto3

        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, settings.AWS_REGION, "aoss")

        return OpenSearch(
            hosts=[{
                "host": settings.OPENSEARCH_HOST,
                "port": 443
            }],
            http_auth=auth,
            http_compress=True,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

    # ──────────────────────────────────────────────
    # Index Management
    # ──────────────────────────────────────────────

    def ensure_index_exists(self) -> None:
        """
        Create the kNN index if it does not already exist.

        Called once at:
        - App startup (via dependencies.py)
        - Start of ingestion script

        The index mapping defines:
        - 'embedding' field: kNN vector with cosine similarity
        - 'text' field: raw chunk text returned at query time
        - 'metadata' field: source doc name, page number, s3 key etc.
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists. Skipping creation.")
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,                    # enables kNN plugin
                    "knn.algo_param.ef_search": 100 # controls recall vs speed tradeoff , search how many nodes before picking chunk- higher no. means more accurate but less speed
                                                    # 100 is production-safe default
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimension,  # must match Titan output (1536)
                        "method": {
                            "name": "hnsw",             # Hierarchical Navigable Small World
                                                        # industry standard ANN algorithm
                            "space_type": "cosinesimil",# cosine similarity matching
                            "engine": "faiss",          # Facebook AI Similarity Search
                                                        # best performance for dense vectors
                            "parameters": {
                                "ef_construction": 128, # higher = better index quality
                                                        # lower = faster index build
                                "m": 16                 # number of neighbours per node
                                                        # 16 is optimal for 1536-dim vectors
                            }
                        }
                    },
                    "text": {
                        "type": "text"              # raw chunk text, returned in results
                    },
                    "metadata": {
                        "type": "object",           # flexible — stores whatever we pass
                        "enabled": True
                    }
                }
            }
        }

        self.client.indices.create(
            index=self.index_name,
            body=index_body
        )
        logger.info(f"Index '{self.index_name}' created successfully with kNN mapping")

    # ──────────────────────────────────────────────
    # Document Ingestion
    # ──────────────────────────────────────────────

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk insert document chunks with their embeddings into OpenSearch.

        Called by: rag/ingestion/embeddings.py after embedding each chunk batch

        Args:
            documents: List of dicts, each must contain:
                {
                    "id":        str,         unique chunk ID  e.g. "doc1_chunk_0"
                    "text":      str,         raw chunk text
                    "embedding": List[float], 1536-dim vector from Titan
                    "metadata":  dict         source_file, page, chunk_index etc.
                }

        Returns:
            { "success": int, "failed": int }
        """
        if not documents:
            logger.warning("upsert_documents called with empty list. Nothing to insert.")
            return {"success": 0, "failed": 0}

        # Build bulk actions in OpenSearch format
        actions = [
            {
                "_op_type": "index",        # 'index' = insert or overwrite
                "_index": self.index_name,
                "_id": doc["id"],           # deterministic ID allows safe re-ingestion
                "_source": {
                    "text": doc["text"],
                    "embedding": doc["embedding"],
                    "metadata": doc.get("metadata", {})
                }
            }
            for doc in documents
        ]

        success, failed = bulk(
            self.client,
            actions,
            raise_on_error=False,       # don't crash on partial failure
            raise_on_exception=False
        )

        logger.info(f"Bulk upsert complete — success: {success}, failed: {len(failed)}")

        if failed:
            logger.error(f"Failed documents: {failed}")

        return {"success": success, "failed": len(failed)}

    # ──────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform kNN similarity search against the vector index.

        Called by: rag/retrieval/retriever.py at query time

        Args:
            query_vector: 1536-dim embedding of the user's question
            top_k:        number of most similar chunks to return (default 5)

        Returns:
            List of dicts, each containing:
            {
                "text":     str,   the chunk text passed to the LLM as context
                "score":    float, cosine similarity score (0–1, higher = more similar)
                "metadata": dict   source file, page, chunk index
            }
        """
        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "_source": ["text", "metadata"]  # only return these fields, not the vector
                                              # returning vectors wastes bandwidth
        }

        response = self.client.search(
            index=self.index_name,
            body=query_body
        )

        hits = response["hits"]["hits"]

        results = [
            {
                "text": hit["_source"]["text"],
                "score": hit["_score"],
                "metadata": hit["_source"].get("metadata", {})
            }
            for hit in hits
        ]

        logger.info(f"kNN search returned {len(results)} results")
        return results

    # ──────────────────────────────────────────────
    # Health Check
    # ──────────────────────────────────────────────

    def health_check(self) -> bool:
        """
        Verify OpenSearch is reachable.
        Called by app startup and monitoring.
        """
        try:
            self.client.cluster.health()
            return True
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {str(e)}")
            return False