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