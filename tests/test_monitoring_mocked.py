from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Mock the expensive resources BEFORE importing main
with patch('backend.storage.vector_store.VectorStore') as MockVectorStore, \
     patch('backend.storage.knowledge_graph.KnowledgeGraph') as MockKG:
    
    # Configure mocks
    mock_vs = MockVectorStore.return_value
    mock_vs.initialize = AsyncMock()
    mock_vs.close = MagicMock()
    
    mock_kg = MockKG.return_value
    mock_kg.close = MagicMock()
    
    # Import app after patching
    from backend.main import app

client = TestClient(app)

def test_metrics_endpoint():
    """Verify that the /metrics endpoint returns Prometheus formatted data"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "http_request_duration_seconds" in response.text

def test_root_metrics_increment():
    """Verify that hitting an endpoint increments metrics"""
    # 1. Hit root
    client.get("/")
    
    # 2. Check metrics
    metrics_after = client.get("/metrics").text
    assert "http_requests_total" in metrics_after
    assert 'endpoint="/"' in metrics_after
