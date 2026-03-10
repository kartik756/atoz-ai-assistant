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

        

