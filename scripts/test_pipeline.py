# scripts/test_embedding.py  ← temporary, deleted after verification

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.bedrock_service import BedrockService

async def main():
    service = BedrockService()
    
    test_text = "What is the leave policy for employees?"
    print(f"\nInput: {test_text}")
    print("Calling Titan Embeddings via Bedrock...")
    
    embedding = await service.embed_text(test_text)
    
    print(f"\nEmbedding dimensions : {len(embedding)}")
    print(f"First 5 values       : {embedding[:5]}")
    print(f"Type                 : {type(embedding[0])}")
    print("\nembed_text() is working correctly.")

asyncio.run(main())