import os
import json
import pytest
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from web_app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_api_fleet_predictions(client):
    response = client.get('/api/fleet/predictions')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert len(data['predictions']) == 500


def test_api_fleet_stats(client):
    response = client.get('/api/fleet/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'network_health_score' in data
    assert 'total_devices' in data


def test_api_fleet_health(client):
    response = client.get('/api/fleet/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'score' in data
    assert 'status' in data


def test_api_devices_list(client):
    response = client.get('/api/devices')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert len(data['devices']) >= 500


def test_api_device_detail(client):
    response = client.get('/api/devices/DEV-0001')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['device']['device_id'] == 'DEV-0001'


def test_api_discovery_scan(client):
    response = client.post('/api/discovery/scan', json={"cidr": "10.1.0.0/24", "mode": "SIMULATION"})
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['discovered_count'] >= 42
