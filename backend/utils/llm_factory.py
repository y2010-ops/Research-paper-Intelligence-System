"""
LLM Factory for creating chat model instances.
Supports Ollama (local) and OpenAI.
"""
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from loguru import logger
from backend.config import settings

def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the configured LLM instance (Ollama or OpenAI).
    
    Args:
        temperature: Model temperature (0.0 for deterministic, 0.7 for creative)
        
    Returns:
        LangChain Chat Model
    """
    if settings.LLM_PROVIDER == "ollama":
        logger.info(f"Using Ollama LLM: {settings.OLLAMA_MODEL}")
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temperature
        )
    
    elif settings.LLM_PROVIDER == "openai":
        logger.info(f"Using OpenAI LLM: {settings.OPENAI_MODEL}")
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature
        )
        
    else:
        # Fallback to Ollama if unknown
        logger.warning(f"Unknown LLM provider '{settings.LLM_PROVIDER}', falling back to Ollama")
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temperature
        )
