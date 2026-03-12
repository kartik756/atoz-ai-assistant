from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


class DocumentLoader:
    """
    Responsible for loading documents from disk
    and extracting raw text content.
    """

    def load_pdfs(self, folder_path: str) -> List[Dict]:
        """
        Load all PDF files from a folder.

        Returns:
            List of documents with text and metadata.
        """

        documents = []

        folder = Path(folder_path)

        for pdf_file in folder.glob("*.pdf"):

            reader = PdfReader(pdf_file)

            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": pdf_file.name
                    }
                }
            )

        return documents