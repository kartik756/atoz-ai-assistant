'''
So this file is responsible for:

Talking to AWS Bedrock
It will handle two operations:

1 Retrieve documents from Knowledge Base
2 Generate response using LLM
'''

import boto3
import logging
from typing import List
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class BedrockService:
    """
    Service responsible for interacting with AWS Bedrock.

    Responsibilities:
    - Retrieve documents from Bedrock Knowledge Base
    - Generate responses using Bedrock models
    """

    def __init__(self):

        self.region = settings.AWS_REGION
        self.kb_id = settings.BEDROCK_KNOWLEDGE_BASE_ID
        self.model_id = settings.BEDROCK_MODEL_ID

        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=self.region
        )
        
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region
        )
       
#This is an asynchronous function that takes a text query and returns a list of text chunks retrieved from the knowledge base.
    async def retrieve_from_kb(self, query: str) -> List[str]:
        """
        Retrieve relevant document chunks from Bedrock Knowledge Base
        """

        try:

            logger.info(f"Retrieving documents from KB for query: {query}")

            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={
                    "text": query
                }
            )

            results = response.get("retrievalResults", [])

            documents = []

            for item in results:
                content = item.get("content", {})
                text = content.get("text")

                if text:
                    documents.append(text)

            logger.info(f"Retrieved {len(documents)} documents from KB")

            return documents

        except Exception as e:

            logger.error(f"KB retrieval failed: {str(e)}")

            raise

    async def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate an answer using Bedrock LLM with retrieved context
        """

        try:

            logger.info("Generating response using Bedrock model")

            context_text = "\n\n".join(context)

            prompt = f"""
You are an internal enterprise AI assistant for employees.

Use the following company documents to answer the question.

Context:
{context_text}

Question:
{query}

Instructions:
- Answer only using the provided context
- If the answer is not found in the context, say you don't know
- Keep the answer concise and clear

Answer:
"""

            body = {
                "prompt": prompt,
                "max_tokens_to_sample": 500,
                "temperature": 0.3
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=str(body)
            )

            response_body = response["body"].read().decode()

            logger.info("Bedrock model response generated")

            return response_body

        except Exception as e:

            logger.error(f"Model generation failed: {str(e)}")

            raise
        

