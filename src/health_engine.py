"""
health_engine.py
----------------
Pure-function health-score utilities for NetGuard NOC.
No ML model dependency — uses the same weighted metric formula as the
synthetic data generator so scores are conceptually consistent with labels.

Public API
----------
compute_health_score(telemetry) -> float          0-100, higher = healthier
risk_window_from_probability(prob) -> str          heuristic bucket label
main_causes(telemetry, top_n) -> list[dict]        ranked worst-metric list
build_health_report(telemetry, probability) -> dict full enriched report
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Feature weights (mirrors the synthetic label equation in generate_dataset.py)
# These drive both the health score and the fallback cause-ranking.
# ---------------------------------------------------------------------------
_WEIGHTS: dict[str, tuple[float, float, str]] = {
    # field_name: (weight, normalisation_max, display_label)
    "CPU_Usage":        (0.25, 100.0,  "CPU Usage"),
    "Memory_Usage":     (0.20, 100.0,  "Memory Usage"),
    "Temperature":      (0.20,  90.0,  "Temperature"),
    "Interface_Errors": (0.10, 100.0,  "Interface Errors"),
    "Packet_Loss":      (0.10,  10.0,  "Packet Loss"),
    "Bandwidth_Usage":  (0.10, 100.0,  "Bandwidth Usage"),
    "Log_Errors":       (0.05,  30.0,  "Log Errors"),
}

# ---------------------------------------------------------------------------
# Threshold recommendations keyed by field
# ---------------------------------------------------------------------------
_ACTIONS: dict[str, str] = {
    "CPU_Usage":        "Redistribute process load or upgrade CPU capacity.",
    "Memory_Usage":     "Check for memory leaks; consider a scheduled reboot.",
    "Temperature":      "Inspect cooling fans, clean chassis vents, reduce ambient heat.",
    "Interface_Errors": "Inspect cables and SFP optics for physical layer faults.",
    "Packet_Loss":      "Investigate buffer bloat or link-level duplex mismatches.",
    "Bandwidth_Usage":  "Consider QoS policies or traffic shaping to reduce saturation.",
    "Log_Errors":       "Review syslog buffer for underlying hardware or auth failures.",
}


def compute_health_score(telemetry: dict[str, Any]) -> float:
    """
    Compute a device health score in [0, 100].
    100 = perfectly healthy, 0 = fully degraded.

    The failure-risk score S is computed exactly like the synthetic label
    equation (weights × normalised metrics). Health = (1 − S) × 100.
    """
    risk_score = 0.0
    for field, (weight, norm_max, _) in _WEIGHTS.items():
        raw = float(telemetry.get(field, 0.0))
        risk_score += weight * min(raw / norm_max, 1.0)

    health = (1.0 - min(risk_score, 1.0)) * 100.0
    return round(health, 1)


def risk_window_from_probability(probability: float) -> str:
    """
    Map a [0, 1] failure probability to an estimated failure window.
    These are heuristic buckets — treat as illustrative until a
    survival-analysis model is integrated (Phase-4 roadmap).
    """
    p = probability
    if p < 0.20:
        return "No imminent risk — routine maintenance schedule"
    elif p < 0.40:
        return "Low risk — monitor within 7–14 days"
    elif p < 0.60:
        return "Moderate risk — inspect within 48–72 hours"
    elif p < 0.75:
        return "Elevated risk — intervene within 24 hours"
    elif p < 0.90:
        return "High risk — immediate maintenance required"
    else:
        return "Critical — failure likely within hours"


def main_causes(
    telemetry: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Rank features by their weighted badness contribution.
    Returns a list of dicts sorted descending by contribution, e.g.:
      [{"feature": "CPU Usage", "contribution": 0.24, "raw_value": 96.5, "unit": "%"}, ...]

    Used as a graceful fallback when SHAP is unavailable.
    """
    units = {
        "CPU_Usage": "%",
        "Memory_Usage": "%",
        "Temperature": "°C",
        "Interface_Errors": "errors",
        "Packet_Loss": "%",
        "Bandwidth_Usage": "%",
        "Log_Errors": "log entries",
    }

    contributions = []
    for field, (weight, norm_max, label) in _WEIGHTS.items():
        raw = float(telemetry.get(field, 0.0))
        contribution = weight * min(raw / norm_max, 1.0)
        contributions.append({
            "feature": label,
            "contribution": round(contribution, 4),
            "raw_value": raw,
            "unit": units.get(field, ""),
        })

    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions[:top_n]


def recommended_actions(
    telemetry: dict[str, Any],
    top_causes: list[dict[str, Any]] | None = None,
    probability: float = 0.0,
) -> list[str]:
    """
    Generate a short list of human-readable action items.
    If top_causes are provided (from SHAP or main_causes), uses those
    to pick the most relevant actions; otherwise falls back to threshold rules.
    """
    actions: list[str] = []

    if top_causes:
        # Map display label back to field key
        label_to_field = {label: k for k, (_, _, label) in _WEIGHTS.items()}
        for cause in top_causes[:3]:
            field = label_to_field.get(cause["feature"])
            if field and _ACTIONS.get(field):
                action = _ACTIONS[field]
                if action not in actions:
                    actions.append(action)

    # Always include threshold-triggered advisories
    cpu = float(telemetry.get("CPU_Usage", 0))
    mem = float(telemetry.get("Memory_Usage", 0))
    temp = float(telemetry.get("Temperature", 0))
    iface = float(telemetry.get("Interface_Errors", 0))
    loss = float(telemetry.get("Packet_Loss", 0))
    log_e = float(telemetry.get("Log_Errors", 0))
    uptime = float(telemetry.get("Uptime", 0))

    if cpu > 85 and _ACTIONS["CPU_Usage"] not in actions:
        actions.append(_ACTIONS["CPU_Usage"])
    if mem > 90 and _ACTIONS["Memory_Usage"] not in actions:
        actions.append(_ACTIONS["Memory_Usage"])
    if temp > 75 and _ACTIONS["Temperature"] not in actions:
        actions.append(_ACTIONS["Temperature"])
    if iface > 100 and _ACTIONS["Interface_Errors"] not in actions:
        actions.append(_ACTIONS["Interface_Errors"])
    if loss > 3 and _ACTIONS["Packet_Loss"] not in actions:
        actions.append(_ACTIONS["Packet_Loss"])
    if log_e > 15 and _ACTIONS["Log_Errors"] not in actions:
        actions.append(_ACTIONS["Log_Errors"])
    if uptime > 365:
        msg = "Device uptime exceeds 1 year — schedule a preventive reboot."
        if msg not in actions:
            actions.append(msg)

    if not actions:
        actions.append("All telemetry lines are within standard parameters. No action required.")

    return actions


def build_health_report(
    telemetry: dict[str, Any],
    probability: float,
    shap_causes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a complete enriched health report dict combining all engine outputs.
    Pass shap_causes from shap_explainer.py when available; otherwise the
    engine falls back to main_causes() for cause ranking.
    """
    health_score = compute_health_score(telemetry)
    risk_window = risk_window_from_probability(probability)

    causes = shap_causes if shap_causes else main_causes(telemetry)
    actions = recommended_actions(telemetry, causes, probability)

    return {
        "health_score": health_score,
        "risk_window": risk_window,
        "top_causes": causes,
        "recommended_actions": actions,
    }
