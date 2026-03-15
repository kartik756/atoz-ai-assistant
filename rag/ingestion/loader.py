"""
Document Loader — S3 Ingestion Layer

Responsibilities:
- List all documents in the S3 bucket
- Download each document
- Extract raw text based on file type (PDF, TXT)
- Return structured document objects for the chunking pipeline

This is the ONLY file that talks to S3 during ingestion.
"""

import boto3
import logging
import io
from typing import List, Dict, Any

import pypdf

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentLoader:
    """
    Downloads and parses documents from S3 into raw text.

    Supported formats:
        - PDF  → extracted via pypdf
        - TXT  → decoded as UTF-8

    Returns structured document dicts for downstream chunking.
    """

    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    def load_from_s3(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List and download all supported documents from S3 bucket.

        Called by: scripts/ingest_documents.py

        Args:
            prefix: S3 key prefix to filter documents e.g. "hr-policies/"
                    Empty string = load everything in the bucket

        Returns:
            List of document dicts:
            {
                "filename": str,   original file name  e.g. "leave_policy.pdf"
                "s3_key":   str,   full S3 key         e.g. "hr-policies/leave_policy.pdf"
                "text":     str,   full extracted text
                "source":   str,   S3 URI              e.g. "s3://bucket/key"
            }
        """
        logger.info(f"Listing documents in s3://{self.bucket_name}/{prefix}")

        # List all objects under the prefix
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix
        )

        documents = []

        for page in pages:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]

                # Skip folder placeholders and unsupported types
                if not self._is_supported(s3_key):
                    logger.debug(f"Skipping unsupported file: {s3_key}")
                    continue

                doc = self._download_and_parse(s3_key)

                if doc:
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from S3")
        return documents

    def _download_and_parse(self, s3_key: str) -> Dict[str, Any] | None:
        """
        Download a single S3 object and extract its text.

        Args:
            s3_key: Full S3 object key

        Returns:
            Document dict or None if parsing fails
        """
        try:
            logger.info(f"Downloading: {s3_key}")

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

            file_bytes = response["Body"].read()
            filename = s3_key.split("/")[-1]

            # Route to correct parser based on extension
            if s3_key.lower().endswith(".pdf"):
                text = self._parse_pdf(file_bytes)
            elif s3_key.lower().endswith(".txt"):
                text = self._parse_txt(file_bytes)
            else:
                return None

            if not text.strip():
                logger.warning(f"Empty text extracted from {s3_key}. Skipping.")
                return None

            return {
                "filename": filename,
                "s3_key": s3_key,
                "text": text,
                "source": f"s3://{self.bucket_name}/{s3_key}"
            }

        except Exception as e:
            logger.error(f"Failed to load {s3_key}: {str(e)}")
            return None             # don't crash entire pipeline on one bad file

    def _parse_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using pypdf.
        Joins all pages with newline separator.
        """
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))

        pages_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        return "\n".join(pages_text)

    def _parse_txt(self, file_bytes: bytes) -> str:
        """
        Decode plain text bytes as UTF-8.
        Falls back to latin-1 if UTF-8 fails (handles legacy docs).
        """
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed, falling back to latin-1")
            return file_bytes.decode("latin-1")

    def _is_supported(self, s3_key: str) -> bool:
        """
        Check if file extension is supported.
        Filters out S3 folder placeholders (keys ending with /).
        """
        supported = (".pdf", ".txt")
        return any(s3_key.lower().endswith(ext) for ext in supported)