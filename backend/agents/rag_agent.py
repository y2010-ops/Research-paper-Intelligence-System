"""
RAG Agent implementation using LangChain.
Supports both OpenAI and Ollama via LLM Factory.
"""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.storage.vector_store import VectorStore
from backend.utils.llm_factory import get_llm

class RAGAgent:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = get_llm(temperature=0.0)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful research assistant. Answer the question based ONLY on the provided context.
        If the answer is not in the context, say "I cannot answer this based on the available documents."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    async def process(self, query: str) -> Dict:
        """
        Process a RAG query.
        
        1. Retrieve relevant chunks
        2. Generate answer
        """
        # 1. Retrieve
        results = await self.vector_store.search(query, top_k=5)
        
        # Format context
        context = "\n\n".join([
            f"[Source: {r['metadata'].get('doc_id', 'unknown')}]\n{r['text']}" 
            for r in results
        ])
        
        # 2. Generate
        answer = await self.chain.ainvoke({
            "context": context,
            "question": query
        })
        
        return {
            "answer": answer,
            "sources": results
        }