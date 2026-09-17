import pytest
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from web_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_stats(client):
    res = client.get('/api/stats')
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

def test_api_predict(client):
    payload = {
        "device_id": "TEST-DEV-01",
        "Device_Type": "Router",
        "CPU_Usage": 85.0,
        "Memory_Usage": 80.0,
        "Temperature": 75.0
    }
    res = client.post('/api/predict', json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "probability" in data
    assert "risk" in data

def test_api_topology(client):
    res = client.get('/api/topology')
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "nodes" in data
