import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Model Providers
    # Options: "openai", "ollama", "local"
    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "local"
    
    # OpenAI Settings (Optional if using Ollama/Local)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_EMBEDDING_MODEL: str = "llama3" # Or use separate embedding model
    
    # Local Embedding Settings
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"
    
    # Vector DB Settings
    VECTOR_DB_TYPE: str = "pinecone" # "pinecone" or "chromadb"
    CHROMADB_PATH: str = "backend/data/chromadb"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: Optional[str] = "us-east-1" # Often not needed for serverless
    PINECONE_INDEX: str = "research-papers"
    
    # Ingestion Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        extra = "ignore" # Allow extra fields in .env

settings = Settings()