import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from discovery import NetworkDiscoveryEngine


def test_discovery_simulation_mode():
    engine = NetworkDiscoveryEngine()
    discovered = engine.scan_subnet("10.1.0.0/24", mode="SIMULATION")
    assert len(discovered) >= 42
    for dev in discovered:
        assert "device_id" in dev
        assert "hostname" in dev
        assert "ip_address" in dev
        assert dev["status"] == "REACHABLE"


def test_discovery_lab_fallback():
    engine = NetworkDiscoveryEngine()
    discovered = engine.scan_subnet("127.0.0.1/32", mode="LAB")
    assert len(discovered) > 0
    assert "ip_address" in discovered[0]
