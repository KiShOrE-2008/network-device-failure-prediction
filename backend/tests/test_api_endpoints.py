import os
import sys
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from web_app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_api_fleet_predictions(client):
    res = client.get('/api/fleet/predictions')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert len(data['predictions']) == 500
    assert 'risk_summary' in data['fleet']


def test_api_fleet_stats(client):
    res = client.get('/api/fleet/stats')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['total_devices'] == 500
    assert 'network_health_score' in data


def test_api_fleet_health(client):
    res = client.get('/api/fleet/health')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'score' in data
    assert data['status'] in ['OPERATIONAL', 'DEGRADED', 'CRITICAL']


def test_api_devices(client):
    res = client.get('/api/devices')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert len(data['devices']) == 500


def test_api_device_detail(client):
    res = client.get('/api/devices/DEV-00001')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['device']['device_id'] == 'DEV-00001'
    assert 'telemetry' in data['device']


def test_api_alerts(client):
    res = client.get('/api/alerts')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert isinstance(data['alerts'], list)


def test_api_alert_acknowledge_and_resolve(client):
    # Fetch active alerts
    res = client.get('/api/alerts')
    data = res.get_json()
    alerts = data['alerts']
    
    if len(alerts) > 0:
        alert_id = alerts[0]['id']
        ack_res = client.post(f'/api/alerts/{alert_id}/acknowledge')
        assert ack_res.status_code == 200
        assert ack_res.get_json()['success'] is True

        res_res = client.post(f'/api/alerts/{alert_id}/resolve')
        assert res_res.status_code == 200
        assert res_res.get_json()['success'] is True


def test_api_discovery_scan_and_nodes(client):
    res = client.post('/api/discovery/scan', json={"cidr": "10.1.0.0/24", "mode": "SIMULATION"})
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['discovered_count'] > 0

    nodes_res = client.get('/api/discovery/nodes')
    assert nodes_res.status_code == 200
    assert nodes_res.get_json()['success'] is True


def test_api_topology(client):
    res = client.get('/api/topology')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'nodes' in data
    assert 'links' in data
