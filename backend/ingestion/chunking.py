from typing import List, Dict, Optional
# Updated import for modern LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from backend.config import settings

class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks"""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
        return chunks

# Singleton
chunker = Chunker()

def chunk_pdf_with_langchain(file_path: str, custom_metadata: Dict = None) -> List[Document]:
    """Load and chunk a PDF"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add metadata
        if custom_metadata:
            for doc in documents:
                doc.metadata.update(custom_metadata)
                
        # Chunk
        return chunker.chunk_documents(documents)
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {e}")
        return []
