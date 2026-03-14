Write-Host "Research Paper Intelligence System - Setup Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "python command not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "Virtual environment created" -ForegroundColor Green

$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

# Step 3: Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip
& $venvPip install -r requirements.txt
Write-Host "Dependencies installed" -ForegroundColor Green

# Step 4: Download spaCy model
Write-Host "Downloading spaCy model..." -ForegroundColor Yellow
& $venvPython -m spacy download en_core_web_sm
Write-Host "spaCy model downloaded" -ForegroundColor Green

# Step 5: Setup environment file
Write-Host "Setting up environment..." -ForegroundColor Yellow
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host ".env file created" -ForegroundColor Green
    Write-Host "Please edit .env and add your API keys" -ForegroundColor Yellow
} else {
    Write-Host ".env file already exists" -ForegroundColor Green
}

# Step 6: Create data directories
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data/papers" | Out-Null
New-Item -ItemType Directory -Force -Path "data/chromadb" | Out-Null
Write-Host "Data directories created" -ForegroundColor Green

# Step 7: Start Neo4j with Docker
Write-Host "Starting Neo4j..." -ForegroundColor Yellow
if (Get-Command docker -ErrorAction SilentlyContinue) {
    try {
        docker-compose up -d neo4j
        Write-Host "Neo4j started (if defined in docker-compose.yml)" -ForegroundColor Green
    } catch {
        Write-Host "Could not run docker-compose. Make sure docker-compose.yml exists." -ForegroundColor Yellow
    }
} else {
    Write-Host "Docker not found. Install Docker to run Neo4j" -ForegroundColor Yellow
}

# Step 8: Download sample papers
Write-Host "Downloading sample papers..." -ForegroundColor Yellow
$samplePapers = @(
    "# Sample AI Research Papers (download these manually)",
    "https://arxiv.org/pdf/1706.03762.pdf  # Attention Is All You Need (Transformers)",
    "https://arxiv.org/pdf/1810.04805.pdf  # BERT",
    "https://arxiv.org/pdf/2005.14165.pdf  # GPT-3",
    "https://arxiv.org/pdf/2103.00020.pdf  # CLIP",
    "https://arxiv.org/pdf/1904.08779.pdf  # DistilBERT"
)
$samplePapers | Out-File -FilePath "data/sample_papers.txt" -Encoding UTF8
Write-Host "Sample paper list created" -ForegroundColor Green
Write-Host "   Download PDFs from links in data/sample_papers.txt"

# Step 9: Verify setup
Write-Host "Verifying setup..." -ForegroundColor Yellow
$verifyScript = "
import langchain
import langgraph
import pinecone
import chromadb
import neo4j
import openai
print('All core libraries imported successfully')
"
& $venvPython -c $verifyScript

Write-Host "==================================================" -ForegroundColor Green
Write-Host "Setup Complete!"
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Edit .env file and add your OpenAI API key"
Write-Host "2. Download sample papers to data/papers/"
Write-Host "3. Run: .\venv\Scripts\python.exe backend/main.py"
Write-Host "4. Access API docs at: http://localhost:8000/docs"
Write-Host ""
