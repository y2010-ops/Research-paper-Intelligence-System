<p align="center">
  <h1 align="center">🧠 Research Paper Intelligence System</h1>
  <p align="center">
    <strong>An AI-powered system for ingesting, analyzing, and querying research papers using RAG and Knowledge Graphs</strong>
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#project-structure">Project Structure</a>
  </p>
</p>

---

## ✨ Features

- **📄 PDF Ingestion Pipeline** — Upload research papers and automatically extract text, chunk content, generate embeddings, and build a knowledge graph.
- **🔍 Hybrid Retrieval** — Combines vector similarity search (RAG) with graph-based relational queries for comprehensive answers.
- **🤖 Multi-Agent Orchestration** — LangGraph-powered workflow that classifies queries and routes them to the optimal search strategy (factual, relational, or hybrid).
- **🕸️ Knowledge Graph** — Automatically extracts entities (authors, concepts, methods) and builds a Neo4j graph to capture relationships between papers.
- **💬 Chat Interface** — Streamlit-based frontend with conversation history, source attribution, and expandable RAG/Graph context.
- **📊 Monitoring & Observability** — Prometheus metrics for latency/throughput tracking and LangSmith integration for LLM tracing and user feedback.
- **🔌 Flexible Model Support** — Supports OpenAI, Ollama (local LLMs), and local sentence-transformer embeddings — switch via configuration.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                          │
│                    (Chat Interface)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│                     FastAPI Backend                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              LangGraph Query Orchestrator                  │ │
│  │                                                            │ │
│  │   Classify ──┬── Factual ──────► RAG Agent ──┐             │ │
│  │   Query      ├── Relational ──► Graph Agent ──┼► Synthesize│ │
│  │              └── Hybrid ──────► Both Agents ──┘             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Ingestion Engine  │  │   Vector Store   │  │ Knowledge     │  │
│  │ • PDF Processing  │  │ • Pinecone       │  │ Graph (Neo4j) │  │
│  │ • Chunking        │  │ • ChromaDB       │  │ • Entities    │  │
│  │ • Embeddings      │  │                  │  │ • Relations   │  │
│  │ • Entity Extract  │  │                  │  │               │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │           Monitoring: Prometheus + LangSmith                ││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Query Flow

1. **Classify** — The LLM classifies the incoming query as `factual`, `relational`, or `hybrid`.
2. **Search** — Routes to the appropriate agent(s):
   - **RAG Agent** — Semantic search over paper chunks via vector store.
   - **Graph Agent** — Cypher queries over the Neo4j knowledge graph.
   - **Hybrid** — Runs both agents in parallel.
3. **Synthesize** — Merges results from all agents into a coherent, cited answer.

---

## 🛠️ Tech Stack

| Layer               | Technology                                         |
|---------------------|----------------------------------------------------|
| **Backend**         | FastAPI, Uvicorn, Pydantic                         |
| **Orchestration**   | LangChain, LangGraph                               |
| **LLM Providers**   | OpenAI (GPT-4), Ollama (Llama 3)                  |
| **Embeddings**      | Sentence-Transformers (`all-MiniLM-L6-v2`), OpenAI |
| **Vector Store**    | Pinecone, ChromaDB                                 |
| **Knowledge Graph** | Neo4j                                               |
| **NLP**             | spaCy (Entity Extraction)                           |
| **PDF Processing**  | PyPDF2, pdfplumber, PyMuPDF                         |
| **Frontend**        | Streamlit                                           |
| **Monitoring**      | Prometheus, LangSmith                               |
| **Containerization**| Docker, Docker Compose                              |
| **Testing**         | Pytest, HTTPX                                       |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.11+
- **Docker** (for Neo4j)
- **Ollama** (optional — for local LLMs) or an OpenAI API key

### Quick Setup

```powershell
# Clone the repository
git clone https://github.com/y2010-ops/Research-paper-Intelligence-System.git
cd Research-paper-Intelligence-System

# Run the automated setup script (Windows)
.\setup.ps1
```

The setup script will:
1. Create a Python virtual environment
2. Install all dependencies
3. Download the spaCy language model
4. Create a `.env` file from `.env.example`
5. Start Neo4j via Docker
6. Generate a sample papers list

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy environment config
cp .env.example .env
# Edit .env with your API keys

# Start Neo4j
docker-compose up -d

# Run the backend
uvicorn backend.main:app --reload --port 8000

# Run the frontend (in a separate terminal)
streamlit run frontend/app.py
```

### Environment Variables

Edit `.env` with your configuration:

```env
# LLM Provider — "openai" or "ollama"
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=local

# OpenAI (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# Vector Store — "pinecone" or "chromadb"
VECTOR_DB_TYPE=chromadb
PINECONE_API_KEY=your_pinecone_api_key_here
```

---

## 📡 API Reference

Once the backend is running, full interactive docs are available at **http://localhost:8000/docs** (Swagger UI).

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| `GET`  | `/`                | System status and active LLM/Embeddings  |
| `POST` | `/ingest/upload`   | Upload a PDF for ingestion               |
| `POST` | `/query`           | Query the knowledge base                 |
| `POST` | `/feedback`        | Submit user feedback (LangSmith)         |
| `GET`  | `/graph/stats`     | Get knowledge graph node/edge counts     |
| `GET`  | `/metrics`         | Prometheus metrics endpoint              |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Transformer architecture?"}'
```

### Example: Upload a Paper

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@path/to/paper.pdf"
```

---

## 📁 Project Structure

```
RPAI_System/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph multi-agent workflow
│   │   ├── rag_agent.py         # RAG retrieval agent
│   │   └── graph_agent.py       # Knowledge graph agent
│   ├── ingestion/
│   │   ├── pdf_processor.py     # PDF text extraction
│   │   ├── chunking.py          # Document chunking strategies
│   │   ├── embeddings.py        # Embedding generation
│   │   └── entity_extractor.py  # NLP entity extraction
│   ├── storage/
│   │   ├── vector_store.py      # Pinecone/ChromaDB interface
│   │   ├── knowledge_graph.py   # Neo4j graph operations
│   │   └── hybrid_search.py     # Hybrid search strategies
│   ├── utils/
│   │   ├── llm_factory.py       # LLM provider factory
│   │   ├── logger.py            # Loguru configuration
│   │   └── metrics.py           # Custom metrics
│   ├── config.py                # Pydantic settings
│   └── main.py                  # FastAPI application entry point
├── frontend/
│   └── app.py                   # Streamlit chat interface
├── tests/
│   ├── test_agents.py           # Agent tests
│   ├── test_ingestion.py        # Ingestion pipeline tests
│   ├── test_retrieval.py        # Retrieval tests
│   └── test_monitoring.py       # Monitoring tests
├── data/
│   └── papers/                  # Uploaded research papers
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Neo4j service definition
├── requirements.txt             # Python dependencies
├── setup.ps1                    # Automated setup script
├── .env.example                 # Environment template
└── README.md
```

---

## 🧪 Running Tests

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_agents.py -v
```

---

## 🐳 Docker

```bash
# Start all services (Neo4j + App)
docker-compose up -d

# Or build and run the app separately
docker build -t rpai-system .
docker run -p 8000:8000 rpai-system
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
