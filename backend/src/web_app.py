"""
src/web_app.py
--------------
NetGuard NOC Web Application Server.
Initializes Flask, static SPA asset serving, registers modular API blueprints,
and provides utility endpoints for manual inference, what-if simulations, and plots.
"""

import os
import sys
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import health_engine
import shap_explainer
import history_store
import anomaly_detection
from intelligence import diagnostic_engine
from monitoring import syslog_collector

FRONTEND_DIR = os.path.abspath(os.path.join(WORKSPACE_ROOT, '..', 'frontend'))
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

# Import and Register Modular Blueprints
from api.fleet import fleet_bp
from api.devices import devices_bp
from api.alerts import alerts_bp
from api.topology import topology_bp
from api.discovery import discovery_bp

app.register_blueprint(fleet_bp)
app.register_blueprint(devices_bp)
app.register_blueprint(alerts_bp)
app.register_blueprint(topology_bp)
app.register_blueprint(discovery_bp)

history_store.init_db()
history_store.seed_devices_from_dataset()

MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models', 'failure_model.pkl')
DIAGNOSTIC_MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models', 'diagnostic_model.pkl')
ANOMALY_MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models', 'anomaly_model.pkl')

failure_model = None
diagnostic_model = None
anomaly_model = None

try:
    if os.path.exists(MODEL_PATH):
        failure_model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded binary failure model from {MODEL_PATH}")
    if os.path.exists(DIAGNOSTIC_MODEL_PATH):
        diagnostic_model = joblib.load(DIAGNOSTIC_MODEL_PATH)
        print(f"✅ Loaded multi-class diagnostic model from {DIAGNOSTIC_MODEL_PATH}")
    if os.path.exists(ANOMALY_MODEL_PATH):
        anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        print(f"✅ Loaded Isolation Forest anomaly model from {ANOMALY_MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Warning loading models: {str(e)}")


def get_active_model_name():
    if failure_model is None:
        return "None (Model not trained)"
    try:
        if hasattr(failure_model, 'named_steps') and 'model' in failure_model.named_steps:
            clf_class = failure_model.named_steps['model'].__class__.__name__
            return clf_class
        return failure_model.__class__.__name__
    except Exception:
        return "Machine Learning Classifier"


def _run_inference(telemetry: dict, log_to_history: bool = True, device_id: str = "manual") -> dict:
    device_type = str(telemetry.get("Device_Type", "Router")).strip()
    device_type = "Router" if device_type.lower() == "router" else "Switch"

    cpu_usage        = float(telemetry.get("CPU_Usage", 0.0))
    memory_usage     = float(telemetry.get("Memory_Usage", 0.0))
    temperature      = float(telemetry.get("Temperature", 0.0))
    uptime           = float(telemetry.get("Uptime", 0.0))
    interface_errors = int(telemetry.get("Interface_Errors", 0))
    packet_loss      = float(telemetry.get("Packet_Loss", 0.0))
    bandwidth_usage  = float(telemetry.get("Bandwidth_Usage", 0.0))
    log_errors       = int(telemetry.get("Log_Errors", 0))

    cpu_trend        = float(telemetry.get("CPU_Trend", 0.0))
    memory_trend     = float(telemetry.get("Memory_Trend", 0.0))
    temp_trend       = float(telemetry.get("Temperature_Trend", 0.0))
    error_trend      = float(telemetry.get("Error_Trend", 0.0))

    cpu_spike        = int(telemetry.get("CPU_Spike", 1 if (cpu_usage > 85 and cpu_trend > 15) else 0))
    temp_spike       = int(telemetry.get("Temperature_Spike", 1 if (temperature > 75 and temp_trend > 5) else 0))
    error_spike      = int(telemetry.get("Error_Spike", 1 if (interface_errors > 20 and error_trend > 10) else 0))

    cpu_5step_avg    = float(telemetry.get("CPU_5step_avg", cpu_usage))
    mem_5step_avg    = float(telemetry.get("Memory_5step_avg", memory_usage))
    temp_5step_avg   = float(telemetry.get("Temperature_5step_avg", temperature))
    err_5step_avg    = float(telemetry.get("Error_5step_avg", float(interface_errors)))
    loss_5step_avg   = float(telemetry.get("PacketLoss_5step_avg", packet_loss))
    loss_trend       = float(telemetry.get("PacketLoss_Trend", 0.0))
    syslog_crit      = int(telemetry.get("Syslog_Critical_Count", 0))

    clean_telemetry = {
        "Device_Type":            device_type,
        "CPU_Usage":              cpu_usage,
        "Memory_Usage":           memory_usage,
        "Temperature":            temperature,
        "Uptime":                 uptime,
        "Interface_Errors":       interface_errors,
        "Packet_Loss":            packet_loss,
        "Bandwidth_Usage":        bandwidth_usage,
        "Log_Errors":             log_errors,
        "Syslog_Critical_Count":  syslog_crit,
        "CPU_5step_avg":          cpu_5step_avg,
        "CPU_Trend":              cpu_trend,
        "CPU_Spike":              cpu_spike,
        "Memory_5step_avg":       mem_5step_avg,
        "Memory_Trend":           memory_trend,
        "Temperature_5step_avg":  temp_5step_avg,
        "Temperature_Trend":      temp_trend,
        "Temperature_Spike":      temp_spike,
        "Error_5step_avg":        err_5step_avg,
        "Error_Trend":            error_trend,
        "Error_Spike":            error_spike,
        "PacketLoss_5step_avg":   loss_5step_avg,
        "PacketLoss_Trend":       loss_trend
    }

    feat_df = pd.DataFrame([clean_telemetry])

    if failure_model is not None:
        try:
            proba = float(failure_model.predict_proba(feat_df)[0, 1])
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            proba = float(min(max((cpu_usage/200.0) + (temperature/180.0) + (interface_errors/50.0), 0.0), 1.0))
    else:
        proba = float(min(max((cpu_usage/200.0) + (temperature/180.0) + (interface_errors/50.0), 0.0), 1.0))

    # Anomaly Detection
    anomaly_score = 0.0
    is_anomaly = False
    if anomaly_model is not None:
        try:
            num_cols = ['CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors', 'Packet_Loss', 'Bandwidth_Usage']
            raw_score = float(anomaly_model.score_samples(feat_df[num_cols])[0])
            anomaly_score = round(float(min(max((-raw_score - 0.3) * 200, 0), 100)), 1)
            is_anomaly = bool(anomaly_model.predict(feat_df[num_cols])[0] == -1)
        except Exception as e:
            print(f"⚠️ Anomaly detection error: {e}")

    # Health & Diagnostics
    risk_level = health_engine.get_risk_level(proba)
    health_score = health_engine.compute_health_score(clean_telemetry, probability=proba, anomaly_score=anomaly_score)

    diag_info = diagnostic_engine.diagnostic_engine.diagnose(clean_telemetry, failure_probability=proba)
    log_info  = syslog_collector.syslog_collector.ingest_and_summarize(clean_telemetry, proba)
    shap_causes = shap_explainer.explain_prediction(clean_telemetry)

    report = health_engine.build_health_report(
        clean_telemetry,
        probability=proba,
        shap_causes=shap_causes,
        anomaly_score=anomaly_score
    )

    result = {
        "failure_probability":     round(proba, 4),
        "failure_probability_pct": round(proba * 100, 1),
        "risk":                    risk_level,
        "risk_window":             health_engine.risk_window_from_probability(proba),
        "predicted_failure":       diag_info["failure_type"],
        "diagnostic_confidence":   diag_info["diagnostic_confidence"],
        "anomaly_score":           anomaly_score,
        "is_anomaly":              is_anomaly,
        "health_score":            health_score,
        "model_name":              get_active_model_name(),
        "recommended_actions":     diag_info["recommended_actions"],
        "top_contributing_causes": shap_causes,
        "health_report":           report,
        "syslog_summary":          log_info
    }

    if log_to_history:
        history_store.log_prediction(device_id, clean_telemetry, result)

    return result


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/v1/health', methods=['GET'])
def api_v1_health():
    return jsonify({
        "status": "HEALTHY",
        "service": "NetGuard NOC Predictive Intelligence Backend",
        "model_status": "AVAILABLE" if failure_model is not None else "UNAVAILABLE"
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        device_id = str(data.get("device_id", "manual")).strip() or "manual"
        result = _run_inference(data, log_to_history=True, device_id=device_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to perform prediction: {str(e)}"}), 500


@app.route('/api/whatif', methods=['POST'])
def whatif():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        result = _run_inference(data, log_to_history=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"What-if inference failed: {str(e)}"}), 500


@app.route('/api/history/<device_id>', methods=['GET'])
def get_device_history(device_id):
    try:
        limit = int(request.args.get("limit", 50))
        records = history_store.get_history(device_id, limit=limit)
        return jsonify({"success": True, "device_id": device_id, "records": records})
    except Exception as e:
        return jsonify({"error": f"History fetch failed: {str(e)}"}), 500


@app.route('/api/device/<device_id>/timeline', methods=['GET'])
def get_device_timeline(device_id):
    csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'netguard_noc_dataset_v1', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        return jsonify({"error": "Time-series dataset CSV not found."}), 404
    try:
        df = pd.read_csv(csv_path)
        dev_df = df[df["Device_ID"] == device_id].sort_values("Timestamp").tail(50)
        records = dev_df.to_dict(orient="records")
        return jsonify({"success": True, "device_id": device_id, "timeline": records})
    except Exception as e:
        return jsonify({"error": f"Timeline fetch failed: {str(e)}"}), 500


@app.route('/api/plots/<filename>')
def serve_plot(filename):
    plots_dir = os.path.join(WORKSPACE_ROOT, 'outputs')
    return send_from_directory(plots_dir, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Launching NetGuard NOC web server on http://localhost:{port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
