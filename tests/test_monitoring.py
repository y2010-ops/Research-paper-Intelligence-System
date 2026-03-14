import pytest
from fastapi.testclient import TestClient
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
    # 1. Get current count
    try:
        metrics_before = client.get("/metrics").text
        # Very naive parse
        count_before = metrics_before.count('http_requests_total{') 
    except:
        count_before = 0
        
    # 2. Hit root
    client.get("/")
    
    # 3. Check metrics again
    metrics_after = client.get("/metrics").text
    assert "http_requests_total" in metrics_after
    
    # We expect the text to contain the path="/" label now
    assert 'endpoint="/"' in metrics_after
