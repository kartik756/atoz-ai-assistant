from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # ─── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "AtoZ AI Assistant"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "local"          # local | dev | prod

    # ─── AWS ───────────────────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"

    # ─── Bedrock ───────────────────────────────────────────────────────────
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"
    BEDROCK_KNOWLEDGE_BASE_ID: str
    EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    EMBEDDING_DIMENSION: int = 1024

    # ─── OpenSearch ────────────────────────────────────────────────────────
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_INDEX: str = "atoz-documents"
    OPENSEARCH_USERNAME: str = ""       # empty for local, set in prod
    OPENSEARCH_PASSWORD: str = ""       # empty for local, set in prod

    # ─── S3 ────────────────────────────────────────────────────────────────
    S3_BUCKET_NAME: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def is_local(self) -> bool:
        """True when running on developer machine"""
        return self.ENVIRONMENT == "local"

    @property
    def opensearch_endpoint(self) -> str:
        """Full OpenSearch base URL"""
        return f"http://{self.OPENSEARCH_HOST}:{self.OPENSEARCH_PORT}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()