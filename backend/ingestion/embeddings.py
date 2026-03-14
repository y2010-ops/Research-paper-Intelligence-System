"""
Embedding generation using LangChain framework.
Demonstrates proper use of LangChain embeddings with fallback options.
"""
from typing import List, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain embedding classes
# LangChain embedding classes
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Fallback embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace embeddings not available")

from backend.config import settings


class LangChainEmbeddingGenerator:
    """
    Embedding generator using LangChain's Embeddings interface.
    
    Shows proper LangChain usage:
    - Primary: OpenAIEmbeddings
    - Fallback: HuggingFaceEmbeddings (local)
    - Caching for efficiency
    """
    
    def __init__(self):
        self.primary_embeddings: Optional[Embeddings] = None
        self.fallback_embeddings: Optional[Embeddings] = None
        self.cache = {}
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize LangChain embedding models based on settings"""
        
        # 1. Local Embeddings (SentenceTransformers) - Default/Preferred
        if settings.EMBEDDING_PROVIDER == "local":
            try:
                if not HUGGINGFACE_AVAILABLE:
                    raise ImportError("sentence-transformers not installed")
                
                self.primary_embeddings = HuggingFaceEmbeddings(
                    model_name=settings.LOCAL_EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"Initialized Local embeddings: {settings.LOCAL_EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Local embeddings: {e}")
                # Fallback to OpenAI if configured
                if settings.OPENAI_API_KEY:
                    logger.warning("Falling back to OpenAI embeddings")
                    self._init_openai()

        # 2. OpenAI Embeddings
        elif settings.EMBEDDING_PROVIDER == "openai":
            self._init_openai()
            
        # 3. Ollama Embeddings
        elif settings.EMBEDDING_PROVIDER == "ollama":
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                self.primary_embeddings = OllamaEmbeddings(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_EMBEDDING_MODEL
                )
                logger.info(f"Initialized Ollama embeddings: {settings.OLLAMA_EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama embeddings: {e}")

    def _init_openai(self):
        """Helper to init OpenAI embeddings"""
        try:
            self.primary_embeddings = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                chunk_size=1000
            )
            logger.info(f"Initialized OpenAI embeddings: {settings.OPENAI_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_query(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single query using LangChain.
        
        This is the LangChain way to embed queries.
        Uses embed_query() method (optimized for queries).
        
        Args:
            text: Query text
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector
        """
        # Check cache
        if use_cache and text in self.cache:
            return self.cache[text]
        
        try:
            # Use primary embeddings (OpenAI)
            if self.primary_embeddings:
                embedding = self.primary_embeddings.embed_query(text)
            elif self.fallback_embeddings:
                embedding = self.fallback_embeddings.embed_query(text)
            else:
                raise RuntimeError("No embedding model available")
            
            # Cache result
            if use_cache:
                self.cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            
            # Try fallback
            if self.fallback_embeddings and self.primary_embeddings:
                logger.warning("Using fallback embeddings")
                embedding = self.fallback_embeddings.embed_query(text)
                if use_cache:
                    self.cache[text] = embedding
                return embedding
            
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents using LangChain.
        
        This is the LangChain way to embed documents.
        Uses embed_documents() method (optimized for batches).
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Use primary embeddings (OpenAI)
            if self.primary_embeddings:
                embeddings = self.primary_embeddings.embed_documents(texts)
                logger.info(f"Generated {len(embeddings)} embeddings with OpenAI")
                return embeddings
            
            # Fallback to HuggingFace
            elif self.fallback_embeddings:
                embeddings = self.fallback_embeddings.embed_documents(texts)
                logger.info(f"Generated {len(embeddings)} embeddings with HuggingFace")
                return embeddings
            
            else:
                raise RuntimeError("No embedding model available")
                
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            
            # Try fallback
            if self.fallback_embeddings and self.primary_embeddings:
                logger.warning("Using fallback embeddings for documents")
                return self.fallback_embeddings.embed_documents(texts)
            
            raise
    
    def embed_langchain_documents(
        self, 
        documents: List[Document]
    ) -> List[List[float]]:
        """
        Embed LangChain Document objects.
        
        Extracts text from Documents and generates embeddings.
        This is the proper way to work with LangChain Documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of embedding vectors
        """
        texts = [doc.page_content for doc in documents]
        return self.embed_documents(texts)
    
    # ==================== Convenience Methods ====================
    
    def generate(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Simple interface - single text embedding.
        Backwards compatible with old code.
        """
        return self.embed_query(text, use_cache)
    
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Simple interface - batch embedding.
        Backwards compatible with old code.
        """
        return self.embed_documents(texts)
    
    # ==================== Advanced Features ====================
    
    def embed_with_metadata(
        self,
        documents: List[Document]
    ) -> List[dict]:
        """
        Embed documents and preserve metadata.
        
        Returns list of dicts with embedding and metadata.
        Useful for maintaining document context.
        """
        embeddings = self.embed_langchain_documents(documents)
        
        results = []
        for doc, embedding in zip(documents, embeddings):
            results.append({
                'text': doc.page_content,
                'embedding': embedding,
                'metadata': doc.metadata
            })
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embedding vectors"""
        test_embedding = self.embed_query("test")
        return len(test_embedding)
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")


# ==================== Multiple Embedding Models ====================

class MultiEmbeddingGenerator:
    """
    Manage multiple embedding models for comparison.
    
    For interview: Shows understanding of different embedding strategies.
    """
    
    def __init__(self):
        self.embeddings = {}
        self._init_multiple_models()
    
    def _init_multiple_models(self):
        """Initialize multiple embedding models"""
        
        # OpenAI text-embedding-3-small (default)
        try:
            self.embeddings['openai-small'] = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.OPENAI_API_KEY
            )
        except:
            pass
        
        # OpenAI text-embedding-3-large (higher quality)
        try:
            self.embeddings['openai-large'] = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=settings.OPENAI_API_KEY
            )
        except:
            pass
        
        # OpenAI ada-002 (legacy, but widely used)
        try:
            self.embeddings['openai-ada'] = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=settings.OPENAI_API_KEY
            )
        except:
            pass
        
        # HuggingFace local models
        if HUGGINGFACE_AVAILABLE:
            try:
                self.embeddings['local-minilm'] = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
            except:
                pass
            
            try:
                self.embeddings['local-mpnet'] = HuggingFaceEmbeddings(
                    model_name="all-mpnet-base-v2"
                )
            except:
                pass
        
        logger.info(f"Initialized {len(self.embeddings)} embedding models")
    
    def embed_with_model(
        self,
        text: str,
        model_name: str = 'openai-small'
    ) -> List[float]:
        """Embed text with specific model"""
        if model_name not in self.embeddings:
            raise ValueError(f"Model {model_name} not available")
        
        return self.embeddings[model_name].embed_query(text)
    
    def compare_embeddings(self, text: str) -> dict:
        """
        Generate embeddings with all models for comparison.
        Useful for evaluation and choosing best model.
        """
        results = {}
        
        for name, embedder in self.embeddings.items():
            try:
                embedding = embedder.embed_query(text)
                results[name] = {
                    'embedding': embedding,
                    'dimension': len(embedding)
                }
            except Exception as e:
                logger.error(f"Failed to embed with {name}: {e}")
        
        return results


# ==================== LangChain Integration with Vector Stores ====================

from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore


class LangChainVectorStoreIntegration:
    """
    Shows how to use LangChain embeddings directly with vector stores.
    
    This is the COMPLETE LangChain workflow:
    1. Load documents with LangChain
    2. Chunk with LangChain
    3. Embed with LangChain
    4. Store in vector DB with LangChain
    """
    
    def __init__(self, embeddings: Embeddings = None):
        # Use provided embeddings or fall back to already-configured generator
        self.embeddings = embeddings or embedding_generator.primary_embeddings
    
    def create_chroma_from_documents(
        self,
        documents: List[Document],
        persist_directory: str = None
    ):
        """
        Create ChromaDB vector store from documents using LangChain.
        
        This is the complete LangChain + ChromaDB integration.
        """
        from langchain_community.vectorstores import Chroma
        
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory or settings.CHROMADB_PATH
        )
        
        logger.info(f"Created ChromaDB with {len(documents)} documents")
        return vectorstore
    
    def create_pinecone_from_documents(
        self,
        documents: List[Document],
        index_name: str = None
    ):
        """
        Create Pinecone vector store from documents using LangChain.
        
        This is the complete LangChain + Pinecone integration.
        """
        # Modern Pinecone Usage: No init() needed, relies on env vars or explicit params
        import os
        from langchain_pinecone import PineconeVectorStore
        
        # Ensure env vars are present if not already
        if settings.PINECONE_API_KEY:
            os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        
        index_name = index_name or settings.PINECONE_INDEX
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=index_name
        )
        
        logger.info(f"Created Pinecone index with {len(documents)} documents")
        return vectorstore


# ==================== Singleton Instances ====================

# Global embedding generator using LangChain
embedding_generator = LangChainEmbeddingGenerator()

# Multi-model generator (for comparison)
multi_embedder = MultiEmbeddingGenerator()

# Vector store integration
vector_integration = LangChainVectorStoreIntegration()


# ==================== Helper Functions ====================

def get_langchain_embeddings() -> Embeddings:
    """
    Get LangChain Embeddings object for use in other components.
    
    This returns the already-configured embeddings (local, OpenAI, or Ollama)
    based on the EMBEDDING_PROVIDER setting.
    """
    return embedding_generator.primary_embeddings