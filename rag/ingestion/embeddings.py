from typing import List, Dict
import json
import boto3

from config.settings import get_settings


class EmbeddingGenerator:
    """
    Responsible for generating embeddings
    for text chunks using Amazon Bedrock.
    """

    def __init__(self):

        settings = get_settings()

        self.model_id = "amazon.titan-embed-text-v1"

        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION
        )

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for each chunk.

        Args:
            chunks: List of chunked documents

        Returns:
            List of documents with embeddings
        """

        embedded_documents = []

        for chunk in chunks:

            text = chunk["text"]

            payload = {
                "inputText": text
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())

            embedding = response_body["embedding"]

            embedded_documents.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": chunk["metadata"]
                }
            )

        return embedded_documents