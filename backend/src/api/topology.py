"""
src/api/topology.py
-------------------
REST API endpoints for network topology graph rendering.
"""

import os
import pandas as pd
from flask import Blueprint, jsonify
from fleet_predictor import FleetPredictor

topology_bp = Blueprint("topology", __name__)
predictor = FleetPredictor()


@topology_bp.route("/api/topology", methods=["GET"])
def get_topology():
    """Returns parent-child topology nodes and links for SVG rendering."""
    csv_path = "data/netguard_noc_dataset_v1/topology.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/topology.csv"

    # Get latest risk predictions
    predictions = predictor.predict_all()
    risk_map = {p["device_id"]: p for p in predictions}

    nodes = []
    links = []

    if os.path.exists(csv_path):
        df_top = pd.read_csv(csv_path)

        unique_ids = set(df_top["Device_ID"].dropna()).union(set(df_top["Parent_ID"].dropna()))
        for dev_id in sorted(unique_ids):
            if dev_id == "ROOT":
                continue
            pred = risk_map.get(dev_id, {})
            nodes.append({
                "id": str(dev_id),
                "label": str(pred.get("hostname", dev_id)),
                "type": str(pred.get("device_type", "ROUTER")),
                "risk": str(pred.get("risk", "LOW")),
                "probability": float(pred.get("failure_probability", 0.0)),
                "health_score": float(pred.get("health_score", 100.0)),
                "status": "CRITICAL" if pred.get("risk") in ["HIGH", "CRITICAL"] else "HEALTHY"
            })

        for _, row in df_top.iterrows():
            parent = str(row.get("Parent_ID"))
            child = str(row.get("Device_ID"))
            if parent and child and parent != "ROOT":
                links.append({
                    "source": parent,
                    "target": child,
                    "relationship": str(row.get("Relationship_Type", "UPLINK"))
                })
    else:
        # Fallback hierarchy
        for dev_id, p in risk_map.items():
            nodes.append({
                "id": dev_id,
                "label": p["hostname"],
                "type": p["device_type"],
                "risk": p["risk"],
                "probability": p["failure_probability"],
                "health_score": p["health_score"]
            })

    return jsonify({
        "success": True,
        "total_nodes": len(nodes),
        "total_links": len(links),
        "nodes": nodes,
        "links": links
    })
