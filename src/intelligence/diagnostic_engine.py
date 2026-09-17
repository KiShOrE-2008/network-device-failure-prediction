"""
src/intelligence/diagnostic_engine.py
------------------------------------
Rule-based & ML hybrid Diagnostic Engine for NetGuard NOC.
Provides multi-mode failure diagnosis, root-cause narrative, and targeted remediation steps.
"""

from typing import Dict, Any, List

FAILURE_MODE_DESCRIPTIONS = {
    "THERMAL": "Thermal overload detected. Elevated chassis temperature combined with fan degradation risk.",
    "MEMORY": "Memory exhaustion detected. Creeping memory utilization pattern consistent with system memory leak.",
    "INTERFACE": "Physical interface failure detected. Accumulating CRC errors and packet loss indicate cable or SFP transceiver defect.",
    "CONGESTION": "Network congestion & buffer bloat detected. Bandwidth saturation and high packet loss degrading performance.",
    "HARDWARE": "Intermittent hardware degradation detected. High uptime and backplane/bus log errors indicate component instability.",
    "NONE": "Device operating within normal operational parameters."
}

ACTION_RECOMMENDATIONS = {
    "THERMAL": [
        "Inspect chassis cooling fans and clear dust filters immediately.",
        "Check rack ambient temperature and air circulation flow.",
        "Redistribute CPU-intensive routing processes to alternate node."
    ],
    "MEMORY": [
        "Inspect memory pool breakdown for leaked process buffers.",
        "Schedule a controlled preventive reboot to flush system memory.",
        "Verify firmware patch release notes for memory leak hotfixes."
    ],
    "INTERFACE": [
        "Clean or replace physical fiber SFP transceiver module.",
        "Inspect patch cable connection and check interface duplex/speed settings.",
        "Run digital optical monitoring (DOM) diagnostic tests."
    ],
    "CONGESTION": [
        "Enforce rate-limiting or QoS traffic shaping policies.",
        "Reroute non-critical traffic paths via secondary egress link.",
        "Evaluate interface bandwidth upgrade options."
    ],
    "HARDWARE": [
        "Perform chassis hardware diagnostic self-test.",
        "Prepare hot-standby failover device for immediate swap.",
        "Contact vendor support for replacement chassis module."
    ],
    "NONE": [
        "No corrective action required. Maintain standard NOC monitoring routine."
    ]
}

def diagnose_failure_mode(
    telemetry: Dict[str, Any],
    ml_predicted_type: str = "NONE",
    failure_prob: float = 0.0
) -> Dict[str, Any]:
    """
    Combines ML classification output with deterministic rules for failure diagnosis.
    """
    cpu = float(telemetry.get("CPU_Usage", 0))
    mem = float(telemetry.get("Memory_Usage", 0))
    temp = float(telemetry.get("Temperature", 0))
    iface_err = float(telemetry.get("Interface_Errors", 0))
    loss = float(telemetry.get("Packet_Loss", 0))
    bw = float(telemetry.get("Bandwidth_Usage", 0))
    
    # Rule-based mode evaluation if probability is significant
    rule_mode = "NONE"
    if failure_prob > 0.35 or temp > 75 or mem > 85 or iface_err > 50 or loss > 3.0:
        if temp > 78.0 and cpu > 70.0:
            rule_mode = "THERMAL"
        elif mem > 88.0:
            rule_mode = "MEMORY"
        elif iface_err > 40 or loss > 4.0:
            rule_mode = "INTERFACE"
        elif bw > 88.0 and loss > 2.0:
            rule_mode = "CONGESTION"
        elif float(telemetry.get("Uptime", 0)) > 300 and float(telemetry.get("Log_Errors", 0)) > 15:
            rule_mode = "HARDWARE"

    # Final diagnosed type (prioritize ML prediction if valid, else rule mode)
    final_type = ml_predicted_type if ml_predicted_type in FAILURE_MODE_DESCRIPTIONS and ml_predicted_type != "NONE" else rule_mode
    if failure_prob < 0.30 and temp < 75 and mem < 80:
        final_type = "NONE"

    description = FAILURE_MODE_DESCRIPTIONS.get(final_type, FAILURE_MODE_DESCRIPTIONS["NONE"])
    actions = ACTION_RECOMMENDATIONS.get(final_type, ACTION_RECOMMENDATIONS["NONE"])

    return {
        "diagnosed_failure_type": final_type,
        "description": description,
        "recommended_actions": actions,
        "rule_override_applied": final_type != ml_predicted_type
    }
