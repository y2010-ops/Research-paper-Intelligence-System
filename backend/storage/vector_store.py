"""
Vector Store implementation using LangChain.
Supports Pinecone (primary) and ChromaDB (fallback/legacy).
"""
import os
from typing import List, Dict, Optional
from loguru import logger

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from backend.config import settings
from backend.ingestion.embeddings import get_langchain_embeddings

class VectorStore:
    """
    Unified vector store interface using LangChain.
    Selects provider based on config (PINECONE or CHROMADB).
    """
    def __init__(self):
        self.vectorstore = None
        self.embeddings = get_langchain_embeddings()
        self.db_type = settings.VECTOR_DB_TYPE
        
        logger.info(f"Initializing Vector Store: {self.db_type}")

    async def initialize(self):
        """Initialize connection to vector DB"""
        if self.db_type == "pinecone":
            self._init_pinecone()
        elif self.db_type == "chromadb":
            self._init_chromadb()
        else:
            raise ValueError(f"Unknown vector DB type: {self.db_type}")

    def _init_pinecone(self):
        """Initialize Pinecone using langchain-pinecone"""
        if not settings.PINECONE_API_KEY:
            # Check environment as fallback
            if not os.getenv("PINECONE_API_KEY"):
                logger.error("PINECONE_API_KEY missing in settings or environment")
                raise ValueError("PINECONE_API_KEY is required")
        else:
            os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

        try:
            # Initialize with existing index
            self.vectorstore = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX,
                embedding=self.embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
            logger.info(f"Connected to Pinecone Index: {settings.PINECONE_INDEX}")
        except Exception as e:
            logger.error(f"Pinecone connection failed: {e}")
            raise

    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            self.vectorstore = Chroma(
                collection_name="research_papers",
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMADB_PATH
            )
            logger.info(f"Connected to ChromaDB at {settings.CHROMADB_PATH}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

    async def add_langchain_documents(self, documents: List[Document]):
        """Add LangChain Documents to store"""
        if not self.vectorstore:
            await self.initialize()
            
        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to {self.db_type}")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search similar documents"""
        if not self.vectorstore:
            await self.initialize()
            
        try:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=top_k,
                filter=filters
            )
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 1.0 # LangChain similarity_search doesn't return score by default
                })
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def as_retriever(self, **kwargs):
        """Get LangChain retriever interface"""
        if not self.vectorstore:
            raise RuntimeError("VectorStore not initialized")
        return self.vectorstore.as_retriever(**kwargs)
        
    def close(self):
        """Close connections (if needed)"""
        # Pinecone client relies on HTTP, no persistent socket usually
        pass
