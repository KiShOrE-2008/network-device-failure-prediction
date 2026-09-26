import os
import joblib
import pandas as pd
import numpy as np

FAILURE_MODES = ["NONE", "THERMAL", "MEMORY", "INTERFACE", "CONGESTION", "HARDWARE"]

RECOMMENDED_ACTIONS = {
    "THERMAL": [
        "Inspect chassis cooling fans and airflow vents for dust or blockage",
        "Verify ambient datacenter / rack HVAC operating temperature",
        "Review CPU-intensive control plane processes and throttle background diagnostics",
        "Check rack thermal metrics and schedule emergency maintenance if temp exceeds 85°C"
    ],
    "MEMORY": [
        "Review memory utilization trends and detect process memory leaks",
        "Inspect buffer allocation pools and active BGP/OSPF route tables",
        "Terminate non-critical monitoring agents or reload stale daemon processes",
        "Schedule maintenance reboot or memory module replacement if leak persists"
    ],
    "INTERFACE": [
        "Inspect physical transceiver (SFP/QSFP) optical signal levels and Tx/Rx power",
        "Check physical Ethernet/fiber cable connections and patch panel integrity",
        "Review switch port CRC/FCS error counters and duplex mismatch settings",
        "Replace degraded optical transceiver or swap port interface"
    ],
    "CONGESTION": [
        "Review interface bandwidth utilization and ingress/egress queue drop counters",
        "Inspect top talkers and netflow traffic statistics for micro-bursts",
        "Apply Quality of Service (QoS) rate-limiting or traffic shaping policies",
        "Reroute non-critical traffic over redundant uplink paths"
    ],
    "HARDWARE": [
        "Inspect hardware system error logs and PCIe bus fault alerts",
        "Verify power supply unit (PSU) redundancy and voltage status",
        "Perform diagnostic POST check and verify ASIC sensor operational state",
        "Prepare hot-standby unit and initiate hardware replacement ticket"
    ],
    "NONE": [
        "Device operating within normal nominal parameters",
        "Maintain standard preventive maintenance schedule"
    ]
}


class DiagnosticEngine:
    def __init__(self, model_path: str = "models/diagnostic_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"[DiagnosticEngine] Warning loading model: {e}")
                self.model = None

    def diagnose(self, telemetry: dict, failure_probability: float = 0.0) -> dict:
        """
        Runs multi-class diagnostic prediction when failure risk > 30% or telemetry anomalies exist.
        Returns failure mode, confidence, and recommended NOC actions.
        """
        # Heuristic rules fallback if ML model is unavailable or operational state normal
        cpu = telemetry.get('CPU_Usage', 0.0)
        temp = telemetry.get('Temperature', 0.0)
        mem = telemetry.get('Memory_Usage', 0.0)
        errors = telemetry.get('Interface_Errors', 0.0)
        packet_loss = telemetry.get('Packet_Loss', 0.0)
        bw = telemetry.get('Bandwidth_Usage', 0.0)

        # Default classification
        predicted_mode = "NONE"
        confidence = 0.95

        if failure_probability >= 0.30 or temp > 75 or mem > 85 or cpu > 90 or errors > 15:
            if self.model is not None:
                try:
                    # Construct feature vector
                    feat_cols = [
                        'CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors',
                        'Packet_Loss', 'Bandwidth_Usage', 'Uptime', 'Log_Errors',
                        'Syslog_Critical_Count', 'CPU_5step_avg', 'CPU_Trend', 'CPU_Spike',
                        'Memory_5step_avg', 'Memory_Trend', 'Temperature_5step_avg',
                        'Temperature_Trend', 'Temperature_Spike', 'Error_5step_avg',
                        'Error_Trend', 'Error_Spike', 'PacketLoss_5step_avg', 'PacketLoss_Trend'
                    ]
                    vec = [float(telemetry.get(c, 0.0)) for c in feat_cols]
                    df_vec = pd.DataFrame([vec], columns=feat_cols)
                    probs = self.model.predict_proba(df_vec)[0]
                    classes = self.model.classes_
                    top_idx = int(np.argmax(probs))
                    predicted_mode = str(classes[top_idx])
                    confidence = float(probs[top_idx])
                except Exception as e:
                    predicted_mode = self._heuristic_diagnose(cpu, temp, mem, errors, packet_loss, bw)
            else:
                predicted_mode = self._heuristic_diagnose(cpu, temp, mem, errors, packet_loss, bw)

        actions = RECOMMENDED_ACTIONS.get(predicted_mode, RECOMMENDED_ACTIONS["NONE"])

        return {
            "failure_type": predicted_mode,
            "diagnostic_confidence": round(confidence * 100, 1),
            "recommended_actions": actions
        }

    def _heuristic_diagnose(self, cpu, temp, mem, errors, packet_loss, bw):
        if temp > 75 or (temp > 70 and cpu > 80):
            return "THERMAL"
        elif mem > 85:
            return "MEMORY"
        elif errors > 15 or packet_loss > 5.0:
            return "INTERFACE"
        elif bw > 85 and (cpu > 75 or packet_loss > 2.0):
            return "CONGESTION"
        elif cpu > 90:
            return "HARDWARE"
        return "HARDWARE"
