"""
src/api/discovery.py
--------------------
REST API endpoints for network CIDR scanning & device discovery.
"""

from flask import Blueprint, jsonify, request
from discovery import NetworkDiscoveryEngine
import history_store

discovery_bp = Blueprint("discovery", __name__)
discovery_engine = NetworkDiscoveryEngine(db_store=history_store)


@discovery_bp.route("/api/discovery/scan", methods=["POST"])
def scan_network():
    """Triggers subnet IP range scanning & registers discovered endpoints."""
    data = request.get_json(silent=True) or {}
    cidr = data.get("cidr", "10.1.0.0/24")
    mode = data.get("mode", "SIMULATION")

    discovered = discovery_engine.scan_subnet(cidr=cidr, mode=mode)
    history_store.register_discovered_devices(discovered)

    return jsonify({
        "success": True,
        "cidr": cidr,
        "mode": mode,
        "discovered_count": len(discovered),
        "devices": discovered
    })


@discovery_bp.route("/api/discovery/nodes", methods=["GET"])
def get_discovered_nodes():
    """Returns list of all discovered network endpoints."""
    nodes = history_store.get_discovered_nodes()
    return jsonify({
        "success": True,
        "count": len(nodes),
        "nodes": nodes
    })
