"""
src/api/topology.py
-------------------
REST API endpoints for network topology graph rendering.
"""

import os
import sys
import pandas as pd
from flask import Blueprint, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fleet_predictor import FleetPredictor

topology_bp = Blueprint("topology", __name__)
predictor = FleetPredictor()


@topology_bp.route("/api/topology", methods=["GET"])
def get_topology():
    """Returns parent-child topology nodes and links for SVG rendering."""
    workspace_root = os.path.dirname(BASE_DIR)
    csv_path = os.path.join(workspace_root, "data", "netguard_noc_dataset_v1", "topology.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(workspace_root, "data", "topology.csv")

    predictions = predictor.predict_all()
    risk_map = {p["device_id"]: p for p in predictions}

    nodes = []
    links = []

    if os.path.exists(csv_path):
        df_top = pd.read_csv(csv_path)

        src_col = "Source_Device_ID" if "Source_Device_ID" in df_top.columns else "Parent_ID"
        tgt_col = "Target_Device_ID" if "Target_Device_ID" in df_top.columns else "Device_ID"
        rel_col = "Relationship" if "Relationship" in df_top.columns else "Relationship_Type"

        unique_ids = set()
        if src_col in df_top.columns:
            unique_ids.update(df_top[src_col].dropna().astype(str))
        if tgt_col in df_top.columns:
            unique_ids.update(df_top[tgt_col].dropna().astype(str))

        for dev_id in sorted(unique_ids):
            if dev_id in ["ROOT", "nan"]:
                continue
            pred = risk_map.get(dev_id, {})
            nodes.append({
                "id": str(dev_id),
                "name": str(pred.get("hostname", dev_id)),
                "label": str(pred.get("hostname", dev_id)),
                "type": str(pred.get("device_type", "Router")),
                "vendor": str(pred.get("vendor", "Cisco")),
                "model": str(pred.get("model", "ISR-4331")),
                "location": str(pred.get("location", "DC-1")),
                "risk": str(pred.get("risk", "LOW")),
                "failure_probability": float(pred.get("failure_probability", 0.0)),
                "health_score": float(pred.get("health_score", 100.0)),
                "status": "CRITICAL" if pred.get("risk") in ["HIGH", "CRITICAL"] else "HEALTHY"
            })

        for _, row in df_top.iterrows():
            src = str(row.get(src_col, ""))
            tgt = str(row.get(tgt_col, ""))
            rel = str(row.get(rel_col, "UPLINK"))
            if src and tgt and src != "nan" and tgt != "nan" and src != "ROOT":
                links.append({
                    "source": src,
                    "target": tgt,
                    "relationship": rel
                })
    
    if not nodes:
        for dev_id, p in risk_map.items():
            nodes.append({
                "id": dev_id,
                "name": p["hostname"],
                "label": p["hostname"],
                "type": p["device_type"],
                "vendor": p["vendor"],
                "model": p["model"],
                "location": p["location"],
                "risk": p["risk"],
                "failure_probability": p["failure_probability"],
                "health_score": p["health_score"],
                "status": "CRITICAL" if p["risk"] in ["HIGH", "CRITICAL"] else "HEALTHY"
            })

    return jsonify({
        "success": True,
        "nodes": nodes,
        "links": links
    })
