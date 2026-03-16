"""
FastAPI Dependency Injection — Pipeline Initialization

Responsibilities:
- Initialize all RAG pipelines once at app startup
- Provide pipeline instances to route handlers via FastAPI DI
- Centralize startup health checks

Why dependency injection over module-level globals:
- Failures surface with clear error messages at startup
- Easy to mock in tests — override the dependency
- Single instance shared across all requests (stateless pipelines)
- Clean separation between initialization and request handling
"""

import logging
from functools import lru_cache

from rag.pipelines.kb_rag import KBRAGPipeline
from rag.pipelines.custom_rag import CustomRAGPipeline
from services.vector_service import VectorService

logger = logging.getLogger(__name__)


@lru_cache()
def get_kb_rag_pipeline() -> KBRAGPipeline:
    """
    Initialize and return the KB RAG pipeline singleton.

    lru_cache ensures this is created once and reused.
    FastAPI calls this function — if it raises, startup fails
    with a clear error rather than a cryptic runtime crash.
    """
    logger.info("Initializing KBRAGPipeline")
    return KBRAGPipeline()


@lru_cache()
def get_custom_rag_pipeline() -> CustomRAGPipeline:
    """
    Initialize and return the Custom RAG pipeline singleton.

    Also ensures OpenSearch index exists before the first
    request hits the pipeline.
    """
    logger.info("Initializing CustomRAGPipeline")

    # Ensure OpenSearch index is ready before serving requests
    # This is idempotent — safe to call every startup
    vector_service = VectorService()
    vector_service.ensure_index_exists()

    return CustomRAGPipeline()