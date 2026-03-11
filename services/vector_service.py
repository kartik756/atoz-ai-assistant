from opensearchpy import OpenSearch
from config.settings import get_settings


settings = get_settings()


class VectorService:
    """
    Service responsible for interacting with OpenSearch.
    Handles vector index creation, document indexing, and similarity search.
    """

    def __init__(self):
        self.client = self._create_client()

    def _create_client(self):
        """
        Create OpenSearch client connection.
        """

        client = OpenSearch(
            hosts=[{
                "host": settings.OPENSEARCH_HOST,
                "port": settings.OPENSEARCH_PORT
            }],
            http_compress=True
        )

        return client
    
    def create_index(self):
        """
        Create OpenSearch vector index if it does not exist.
        """

        index_name = settings.OPENSEARCH_INDEX

        if self.client.indices.exists(index=index_name):
            print(f"Index {index_name} already exists")
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1536  #dimension MUST match embedding model
                    },
                    "document_name": {
                        "type": "keyword"
                    },
                    "page_number": {
                        "type": "integer"
                    },
                    "source": {
                        "type": "keyword"
                    }
                }
            }
        }

        self.client.indices.create(
            index=index_name,
            body=index_body
        )

        print(f"Created index: {index_name}")
