import os
import joblib
import pandas as pd
import numpy as np
from feature_engineering import DIAGNOSTIC_FEATURE_COLUMNS, prepare_feature_matrix

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
        candidate_paths = [
            self.model_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.model_path) if not os.path.isabs(self.model_path) else None,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "diagnostic_model.pkl"),
        ]
        for p in candidate_paths:
            if p and os.path.exists(p):
                try:
                    self.model = joblib.load(p)
                    return
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
                    df_vec = prepare_feature_matrix(pd.DataFrame([telemetry]), DIAGNOSTIC_FEATURE_COLUMNS)
                    probs = self.model.predict_proba(df_vec)[0]
                    top_idx = int(np.argmax(probs))
                    if hasattr(self.model, "label_classes_"):
                        predicted_mode = str(self.model.label_classes_[top_idx])
                    else:
                        raw_cls = self.model.classes_[top_idx]
                        if isinstance(raw_cls, (int, np.integer)):
                            predicted_mode = self._heuristic_diagnose(cpu, temp, mem, errors, packet_loss, bw)
                        else:
                            predicted_mode = str(raw_cls)
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

    def diagnose_batch(self, df: pd.DataFrame, failure_probabilities: list = None) -> list:
        """
        Efficient vectorized batch diagnosis across multiple devices.
        Avoids single-row DataFrame construction overhead in large loops.
        """
        n = len(df)
        if n == 0:
            return []

        cpus = df['CPU_Usage'].to_numpy(dtype=float) if 'CPU_Usage' in df.columns else np.zeros(n)
        temps = df['Temperature'].to_numpy(dtype=float) if 'Temperature' in df.columns else np.zeros(n)
        mems = df['Memory_Usage'].to_numpy(dtype=float) if 'Memory_Usage' in df.columns else np.zeros(n)
        errors = df['Interface_Errors'].to_numpy(dtype=float) if 'Interface_Errors' in df.columns else np.zeros(n)
        losses = df['Packet_Loss'].to_numpy(dtype=float) if 'Packet_Loss' in df.columns else np.zeros(n)
        bws = df['Bandwidth_Usage'].to_numpy(dtype=float) if 'Bandwidth_Usage' in df.columns else np.zeros(n)

        probs = np.array([p if p is not None else 0.0 for p in failure_probabilities], dtype=float) if failure_probabilities is not None else np.zeros(n)

        needs_ml = (probs >= 0.30) | (temps > 75) | (mems > 85) | (cpus > 90) | (errors > 15)

        predicted_modes = ["NONE"] * n
        confidences = [95.0] * n

        if self.model is not None and np.any(needs_ml):
            try:
                sub_indices = np.where(needs_ml)[0]
                sub_df = df.iloc[sub_indices]
                df_vec = prepare_feature_matrix(sub_df, DIAGNOSTIC_FEATURE_COLUMNS)
                ml_probs = self.model.predict_proba(df_vec)
                top_indices = np.argmax(ml_probs, axis=1)

                classes = getattr(self.model, "label_classes_", getattr(self.model, "classes_", None))

                for local_i, global_i in enumerate(sub_indices):
                    cls_idx = top_indices[local_i]
                    if classes is not None and not isinstance(classes[cls_idx], (int, np.integer)):
                        mode = str(classes[cls_idx])
                    else:
                        mode = self._heuristic_diagnose(cpus[global_i], temps[global_i], mems[global_i], errors[global_i], losses[global_i], bws[global_i])
                    predicted_modes[global_i] = mode
                    confidences[global_i] = round(float(ml_probs[local_i, cls_idx]) * 100, 1)
            except Exception as e:
                for global_i in np.where(needs_ml)[0]:
                    predicted_modes[global_i] = self._heuristic_diagnose(cpus[global_i], temps[global_i], mems[global_i], errors[global_i], losses[global_i], bws[global_i])
        else:
            for global_i in np.where(needs_ml)[0]:
                predicted_modes[global_i] = self._heuristic_diagnose(cpus[global_i], temps[global_i], mems[global_i], errors[global_i], losses[global_i], bws[global_i])

        results = []
        for i in range(n):
            mode = predicted_modes[i]
            results.append({
                "failure_type": mode,
                "diagnostic_confidence": confidences[i],
                "recommended_actions": RECOMMENDED_ACTIONS.get(mode, RECOMMENDED_ACTIONS["NONE"])
            })
        return results

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


# Singleton instance for modular imports
diagnostic_engine = DiagnosticEngine()


def diagnose_failure_mode(telemetry: dict, ml_type: str = "NONE", probability: float = 0.0) -> dict:
    """Diagnoses failure mode using ML diagnostic model or heuristic fallback."""
    res = diagnostic_engine.diagnose(telemetry, failure_probability=probability)
    predicted_type = ml_type if (ml_type and ml_type != "NONE") else res.get("failure_type", "NONE")
    actions = RECOMMENDED_ACTIONS.get(predicted_type, RECOMMENDED_ACTIONS.get("NONE", []))
    descriptions = {
        "THERMAL": "Chassis thermal overload detected; cooling capacity or fan degradation likely.",
        "MEMORY": "Memory utilization alert; potential memory leak or buffer exhaustion.",
        "INTERFACE": "Physical interface packet drops or transceiver signal degradation.",
        "CONGESTION": "Traffic saturation and queue buffer drops exceeding operational thresholds.",
        "HARDWARE": "Hardware subsystem alert, voltage anomaly or bus faults.",
        "NONE": "Device operating within normal nominal parameters."
    }
    return {
        "diagnosed_failure_type": predicted_type,
        "failure_type": predicted_type,
        "description": descriptions.get(predicted_type, f"Telemetry indicates potential {predicted_type} risk."),
        "recommended_actions": actions,
        "diagnostic_confidence": res.get("diagnostic_confidence", 95.0)
    }

