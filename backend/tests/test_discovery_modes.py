import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from discovery import NetworkDiscoveryEngine


def test_simulation_discovery():
    """Verify SIMULATION mode loads device inventory without network scanning."""
    engine = NetworkDiscoveryEngine()
    nodes = engine.scan_subnet(cidr="10.1.0.0/24", mode="SIMULATION")
    assert len(nodes) >= 2
    assert nodes[0]["discovery_protocol"] == "SIMULATED_INVENTORY" or nodes[0]["discovery_protocol"] == "SIMULATED_SNMP"


def test_lab_discovery():
    """Verify LAB mode performs bounded reachability probing without raising errors."""
    engine = NetworkDiscoveryEngine()
    nodes = engine.scan_subnet(cidr="127.0.0.0/30", mode="LAB")
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_production_mode_is_explicit():
    """Verify PRODUCTION mode requires explicit authenticated credentials and does not fake production SNMP data."""
    engine = NetworkDiscoveryEngine()
    
    # Run in production mode without SNMP_COMMUNITY set
    if "SNMP_COMMUNITY" in os.environ:
        del os.environ["SNMP_COMMUNITY"]

    nodes = engine.scan_subnet(cidr="10.1.0.0/24", mode="PRODUCTION")
    assert len(nodes) > 0
    assert nodes[0]["discovery_protocol"] == "SNMP_V3_REST_API"
    assert "AUTHENTICATED_SNMP_REQUIRED" in nodes[0]["model"] or "SIMULATED" in nodes[0]["discovery_protocol"]


def test_invalid_cidr_validation():
    """Verify engine raises ValueError on invalid CIDR string."""
    engine = NetworkDiscoveryEngine()
    with pytest.raises(ValueError, match="Invalid CIDR"):
        engine.scan_subnet(cidr="invalid_ip_range", mode="SIMULATION")
