"""
src/api/alerts.py
-----------------
REST API endpoints for NOC incident alerts management.
"""

from flask import Blueprint, jsonify, request
import history_store

alerts_bp = Blueprint("alerts", __name__)


@alerts_bp.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Returns active and acknowledged NOC incident alerts."""
    alerts = history_store.get_active_alerts(limit=50)
    return jsonify({
        "success": True,
        "count": len(alerts),
        "alerts": alerts
    })


@alerts_bp.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert_route(alert_id: int):
    """Marks an incident alert as ACKNOWLEDGED."""
    success = history_store.acknowledge_alert(alert_id)
    if success:
        return jsonify({"success": True, "message": f"Alert {alert_id} acknowledged."})
    return jsonify({"success": False, "error": f"Alert {alert_id} not found."}), 404


@alerts_bp.route("/api/alerts/<int:alert_id>/resolve", methods=["POST"])
def resolve_alert_route(alert_id: int):
    """Marks an incident alert as RESOLVED."""
    success = history_store.resolve_alert(alert_id)
    if success:
        return jsonify({"success": True, "message": f"Alert {alert_id} resolved."})
    return jsonify({"success": False, "error": f"Alert {alert_id} not found."}), 404
