from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import shutil
import os
import time
from typing import List, Optional
from loguru import logger
from pydantic import BaseModel

# Monitoring
from prometheus_client import make_asgi_app, Counter, Histogram
from langsmith import Client

from backend.config import settings
from backend.ingestion.pdf_processor import PDFProcessor
from backend.ingestion.chunking import chunker, chunk_pdf_with_langchain
from backend.ingestion.embeddings import embedding_generator
from backend.ingestion.entity_extractor import EntityExtractor
from backend.storage.vector_store import VectorStore
from backend.storage.knowledge_graph import KnowledgeGraph
from backend.agents.orchestrator import QueryOrchestrator

# Global State
vector_store = VectorStore()
knowledge_graph = KnowledgeGraph()
orchestrator = None
ls_client = Client() # LangSmith Client

# Prometheus Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("System startup...")
    await vector_store.initialize()
    global orchestrator
    orchestrator = QueryOrchestrator(vector_store, knowledge_graph)
    
    yield
    
    # Shutdown
    vector_store.close()
    knowledge_graph.close()

app = FastAPI(title="Research Paper Intelligence System", lifespan=lifespan)

# Prometheus Middleware
@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Expose Metrics for Scraping
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/")
def read_root():
    return {"status": "online", "llm": settings.LLM_PROVIDER, "embeddings": settings.EMBEDDING_PROVIDER}

@app.post("/ingest/upload")
async def upload_paper(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and process a research paper PDF.
    """
    file_path = f"backend/data/{file.filename}"
    os.makedirs("backend/data", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_document, file_path)
    return {"message": f"Processing started for {file.filename}"}

async def process_document(file_path: str):
    """
    Full Ingestion Pipeline:
    1. Extract Text
    2. Chunk & Embed -> Vector Store
    3. Extract Entities -> Knowledge Graph
    """
    filename = os.path.basename(file_path)
    logger.info(f"Starting processing for {filename}")
    
    try:
        # 1. Processing (LangChain handles loading + chunking)
        chunks = chunk_pdf_with_langchain(file_path, custom_metadata={"title": filename})
        # Note: chunks are List[Document]
        
        # 2. Vector Store (Embed & Store)
        # LangChainVectorStore handles embedding generation automatically
        await vector_store.add_langchain_documents(chunks)
        
        # 3. Entity Extraction & Knowledge Graph
        # Get full text for extraction context (limit to first 5000 chars for efficiency)
        full_text = " ".join([c.page_content for c in chunks[:5]]) 
        
        extractor = EntityExtractor()
        entities = await extractor.extract_entities(full_text)
        
        paper_data = {
            "title": filename,
            "url": file_path,
            **entities
        }
        
        knowledge_graph.add_paper(paper_data)
        logger.info(f"Completed processing for {filename}")
        
    except Exception as e:
        logger.error(f"Processing failed for {filename}: {e}")

class FeedbackRequest(BaseModel):
    run_id: str
    score: int # 1-5
    comment: Optional[str] = None

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback to LangSmith.
    """
    try:
        ls_client.create_feedback(
            run_id=feedback.run_id,
            key="user_score",
            score=feedback.score,
            comment=feedback.comment
        )
        return {"status": "success", "message": "Feedback submitted"}
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_system(query_request: dict):
    """
    Query the system (RAG + Graph).
    Body: {"query": "..."}
    """
    query = query_request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    
    # Run the query
    result = await orchestrator.process_query(query)
    
    # LangGraph returns the state, but we need the run_id for feedback.
    # Currently LangGraph doesn't easily expose the root run_id in the return value of ainvoke 
    # unless using the streaming interface or callbacks manually.
    # However, since we enabled global tracing via Env Vars, it is being traced.
    # For a production app, we would wrap this in a traceable context to get the ID.
    
    return result

@app.get("/graph/stats")
def graph_stats():
    """Get graph statistics"""
    return {
        "nodes": knowledge_graph.execute_query("MATCH (n) RETURN count(n) as count")[0]['count'],
        "rels": knowledge_graph.execute_query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
    }