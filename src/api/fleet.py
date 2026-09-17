"""
src/api/fleet.py
----------------
REST API endpoints for fleet-wide failure predictions, statistics, and network health.
"""

from flask import Blueprint, jsonify, request
from fleet_predictor import FleetPredictor
import history_store

fleet_bp = Blueprint("fleet", __name__)
predictor = FleetPredictor()


@fleet_bp.route("/api/fleet/predictions", methods=["GET"])
def get_fleet_predictions():
    """Returns predictions for all devices in the network fleet, sorted descending by failure risk."""
    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)
    return jsonify({
        "success": True,
        "fleet": summary,
        "predictions": predictions
    })


@fleet_bp.route("/api/fleet/stats", methods=["GET"])
def get_fleet_stats():
    """Returns high-level NOC fleet statistics."""
    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)
    return jsonify({
        "success": True,
        "total_devices": summary["total_devices"],
        "healthy_devices": summary["risk_summary"]["low"],
        "medium_risk_devices": summary["risk_summary"]["medium"],
        "high_risk_devices": summary["risk_summary"]["high"],
        "critical_devices": summary["risk_summary"]["critical"],
        "network_health_score": summary["network_health_score"],
        "average_failure_probability": summary["average_failure_probability"],
        "predicted_failures_next_12h": summary["predicted_failures_next_12h"],
        "failure_mode_breakdown": summary["failure_mode_breakdown"]
    })


@fleet_bp.route("/api/fleet/health", methods=["GET"])
def get_fleet_health():
    """Returns overall network health score and health status."""
    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)

    score = summary["network_health_score"]
    status = "OPERATIONAL"
    if score < 70:
        status = "CRITICAL"
    elif score < 85:
        status = "DEGRADED"

    return jsonify({
        "success": True,
        "score": score,
        "status": status,
        "total_devices": summary["total_devices"],
        "low": summary["risk_summary"]["low"],
        "medium": summary["risk_summary"]["medium"],
        "high": summary["risk_summary"]["high"],
        "critical": summary["risk_summary"]["critical"]
    })


@fleet_bp.route("/api/fleet/predict", methods=["POST"])
def trigger_batch_predict():
    """Triggers batch inference over all 500+ devices and persists prediction logs."""
    predictions = predictor.predict_all()
    summary = predictor.get_fleet_summary(predictions)

    # Persist predictions to database
    for p in predictions:
        history_store.log_prediction(p["device_id"], p.get("telemetry", {}), p)

    return jsonify({
        "success": True,
        "message": f"Successfully analyzed {len(predictions)} devices.",
        "summary": summary
    })
