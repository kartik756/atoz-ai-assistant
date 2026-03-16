'''

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.s3_service import S3Service
from scripts.ingest_documents import ingest_documents
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

s3_service = S3Service()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    try:

        logger.info(f"Uploading file: {file.filename}")

        # Save temporarily
        temp_path = f"data/documents/{file.filename}"

        os.makedirs("data/documents", exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Upload to S3
        s3_service.upload_file(open(temp_path, "rb"), file.filename)

        # Run ingestion
        ingest_documents("data/documents")

        return {
            "message": "Document uploaded and ingested successfully"
        }

    except Exception as e:

        logger.error(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e) 
        )

'''