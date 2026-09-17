"""
src/health_engine.py
--------------------
Pure-function health-score & risk-window utilities for NetGuard NOC.
Combines failure risk, anomaly index, trend velocity, and metric spikes into a unified Health Score [0-100].

Public API
----------
compute_health_score(telemetry, probability=0.0, anomaly_score=0.0) -> float
get_risk_level(probability) -> str
risk_window_from_probability(prob) -> str
main_causes(telemetry, top_n) -> list[dict]
build_health_report(telemetry, probability, shap_causes, anomaly_score) -> dict
"""

from __future__ import annotations
from typing import Any, Dict, List

_WEIGHTS: Dict[str, tuple[float, float, str]] = {
    "CPU_Usage":        (0.20, 100.0, "CPU Usage"),
    "Memory_Usage":     (0.18, 100.0, "Memory Usage"),
    "Temperature":      (0.18,  90.0, "Temperature"),
    "Interface_Errors": (0.10, 100.0, "Interface Errors"),
    "Packet_Loss":      (0.10,  10.0, "Packet Loss"),
    "Bandwidth_Usage":  (0.08, 100.0, "Bandwidth Usage"),
    "Log_Errors":       (0.04,  30.0, "Log Errors"),
    "CPU_Trend":        (0.04,  25.0, "CPU Trend"),
    "Temperature_Trend":(0.04,  15.0, "Temperature Trend"),
    "Error_Trend":      (0.04,  25.0, "Error Velocity")
}

_ACTIONS: Dict[str, str] = {
    "CPU_Usage":        "Redistribute process load or upgrade CPU capacity.",
    "Memory_Usage":     "Check for memory leaks; consider a scheduled reboot.",
    "Temperature":      "Inspect cooling fans, clean chassis vents, reduce ambient heat.",
    "Interface_Errors": "Inspect cables and SFP optics for physical layer faults.",
    "Packet_Loss":      "Investigate buffer bloat or link-level duplex mismatches.",
    "Bandwidth_Usage":  "Consider QoS policies or traffic shaping to reduce saturation.",
    "Log_Errors":       "Review syslog buffer for underlying hardware or auth failures."
}

def compute_health_score(telemetry: Dict[str, Any], probability: float = 0.0, anomaly_score: float = 0.0) -> float:
    """
    Computes unified Health Score [0, 100]. 100 = perfectly healthy, 0 = critical degradation.
    Integrates telemetry metrics, rolling trends, spike penalties, failure risk, and anomaly index.
    """
    base_risk = 0.0
    for field, (weight, norm_max, _) in _WEIGHTS.items():
        raw = float(telemetry.get(field, 0.0))
        base_risk += weight * min(max(raw, 0.0) / norm_max, 1.0)

    # Spike Penalties
    spike_penalty = 0.0
    if float(telemetry.get("CPU_Spike", 0)) > 0: spike_penalty += 0.05
    if float(telemetry.get("Temperature_Spike", 0)) > 0: spike_penalty += 0.08
    if float(telemetry.get("Error_Spike", 0)) > 0: spike_penalty += 0.05

    # Combine Base Risk (40%), ML Failure Prob (40%), Anomaly Index (10%), Spikes (10%)
    combined_risk = (
        0.35 * base_risk +
        0.45 * min(max(probability, 0.0), 1.0) +
        0.10 * min(max(anomaly_score, 0.0) / 100.0, 1.0) +
        0.10 * min(spike_penalty, 1.0)
    )

    health = (1.0 - min(combined_risk, 1.0)) * 100.0
    return round(health, 1)

def get_risk_level(probability: float) -> str:
    """
    4-Tier Risk Scale:
    0.00 - 0.30 -> LOW
    0.30 - 0.65 -> MEDIUM
    0.65 - 0.85 -> HIGH
    0.85 - 1.00 -> CRITICAL
    """
    p = probability
    if p < 0.30:
        return "LOW"
    elif p < 0.65:
        return "MEDIUM"
    elif p < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"

def risk_window_from_probability(probability: float) -> str:
    p = probability
    if p < 0.30:
        return "No imminent risk — routine maintenance schedule"
    elif p < 0.65:
        return "Low-Moderate risk — monitor within 7–14 days"
    elif p < 0.85:
        return "Elevated risk — inspect & intervene within 24–48 hours"
    else:
        return "CRITICAL — failure imminent within hours"

def main_causes(telemetry: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    units = {
        "CPU_Usage": "%", "Memory_Usage": "%", "Temperature": "°C",
        "Interface_Errors": "errors", "Packet_Loss": "%", "Bandwidth_Usage": "%",
        "Log_Errors": "log entries"
    }
    contributions = []
    for field, (weight, norm_max, label) in _WEIGHTS.items():
        raw = float(telemetry.get(field, 0.0))
        contribution = weight * min(max(raw, 0.0) / norm_max, 1.0)
        contributions.append({
            "feature": label,
            "contribution": round(contribution, 4),
            "raw_value": raw,
            "unit": units.get(field, "")
        })
    contributions.sort(key=lambda x: x["contribution"], reverse=True)
    return contributions[:top_n]

def recommended_actions(telemetry: Dict[str, Any], top_causes: List[Dict[str, Any]] | None = None, probability: float = 0.0) -> List[str]:
    actions = []
    if top_causes:
        label_to_field = {label: k for k, (_, _, label) in _WEIGHTS.items()}
        for cause in top_causes[:3]:
            field = label_to_field.get(cause["feature"])
            if field and _ACTIONS.get(field):
                action = _ACTIONS[field]
                if action not in actions:
                    actions.append(action)

    cpu = float(telemetry.get("CPU_Usage", 0))
    mem = float(telemetry.get("Memory_Usage", 0))
    temp = float(telemetry.get("Temperature", 0))
    iface = float(telemetry.get("Interface_Errors", 0))
    loss = float(telemetry.get("Packet_Loss", 0))

    if cpu > 85 and _ACTIONS["CPU_Usage"] not in actions: actions.append(_ACTIONS["CPU_Usage"])
    if mem > 90 and _ACTIONS["Memory_Usage"] not in actions: actions.append(_ACTIONS["Memory_Usage"])
    if temp > 75 and _ACTIONS["Temperature"] not in actions: actions.append(_ACTIONS["Temperature"])
    if iface > 50 and _ACTIONS["Interface_Errors"] not in actions: actions.append(_ACTIONS["Interface_Errors"])
    if loss > 3.0 and _ACTIONS["Packet_Loss"] not in actions: actions.append(_ACTIONS["Packet_Loss"])

    if not actions:
        actions.append("All telemetry lines operating within standard design parameters.")
    return actions

def build_health_report(telemetry: Dict[str, Any], probability: float, shap_causes: List[Dict[str, Any]] | None = None, anomaly_score: float = 0.0) -> Dict[str, Any]:
    health_score = compute_health_score(telemetry, probability, anomaly_score)
    risk_level = get_risk_level(probability)
    risk_window = risk_window_from_probability(probability)
    causes = shap_causes if shap_causes else main_causes(telemetry)
    actions = recommended_actions(telemetry, causes, probability)

    return {
        "health_score": health_score,
        "risk_level": risk_level,
        "risk_window": risk_window,
        "top_causes": causes,
        "recommended_actions": actions
    }
