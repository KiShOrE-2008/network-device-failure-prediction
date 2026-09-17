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

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='')

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
    except:
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
    loss_trend       = float(telemetry.get("PacketLoss_Trend", 0.0))

    cpu_spike        = int(telemetry.get("CPU_Spike", 1 if cpu_trend > 15 else 0))
    temp_spike       = int(telemetry.get("Temperature_Spike", 1 if temp_trend > 8 else 0))
    error_spike      = int(telemetry.get("Error_Spike", 1 if error_trend > 10 else 0))

    cpu_5step        = float(telemetry.get("CPU_5step_avg", cpu_usage))
    memory_5step     = float(telemetry.get("Memory_5step_avg", memory_usage))
    temp_5step       = float(telemetry.get("Temperature_5step_avg", temperature))
    error_5step      = float(telemetry.get("Error_5step_avg", interface_errors))
    loss_5step       = float(telemetry.get("PacketLoss_5step_avg", packet_loss))

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
        "CPU_5step_avg":          cpu_5step,
        "Memory_5step_avg":       memory_5step,
        "Temperature_5step_avg":  temp_5step,
        "Error_5step_avg":        error_5step,
        "PacketLoss_5step_avg":   loss_5step,
        "CPU_Trend":              cpu_trend,
        "Memory_Trend":           memory_trend,
        "Temperature_Trend":      temp_trend,
        "Error_Trend":            error_trend,
        "PacketLoss_Trend":       loss_trend,
        "CPU_Spike":              cpu_spike,
        "Temperature_Spike":      temp_spike,
        "Error_Spike":            error_spike
    }

    features_df = pd.DataFrame([clean_telemetry])

    if failure_model is not None:
        prediction  = int(failure_model.predict(features_df)[0])
        probability = float(failure_model.predict_proba(features_df)[0][1])
    else:
        probability = min(1.0, (cpu_usage*0.25 + memory_usage*0.2 + temperature*0.2 + interface_errors*0.1) / 100.0)
        prediction  = 1 if probability > 0.65 else 0

    anom_res = anomaly_detection.predict_anomaly(clean_telemetry, anomaly_model, ANOMALY_MODEL_PATH)

    ml_failure_type = "NONE"
    if diagnostic_model is not None and (probability > 0.30 or cpu_usage > 85 or temperature > 80 or interface_errors > 40):
        try:
            pred_idx = int(diagnostic_model.predict(features_df)[0])
            if hasattr(diagnostic_model, 'label_classes_') and pred_idx < len(diagnostic_model.label_classes_):
                ml_failure_type = str(diagnostic_model.label_classes_[pred_idx])
        except Exception:
            ml_failure_type = "NONE"

    diag_res = diagnostic_engine.diagnose_failure_mode(clean_telemetry, ml_failure_type, probability)
    final_failure_type = diag_res["diagnosed_failure_type"]

    risk = health_engine.get_risk_level(probability)
    risk_colors = {
        "LOW": "#00e676",
        "MEDIUM": "#ffb300",
        "HIGH": "#ff9100",
        "CRITICAL": "#ff1744"
    }
    color = risk_colors.get(risk, "#00e676")
    
    status_texts = {
        "LOW": "Device condition is currently healthy and operating within nominal parameters.",
        "MEDIUM": "Moderate degradation detected. Recommend close NOC telemetry monitoring.",
        "HIGH": "High risk of component failure. Inspect physical optics and process queue.",
        "CRITICAL": "CRITICAL THREAT: Failure imminent within hours. Execute immediate failover protocol."
    }
    status_text = status_texts.get(risk, status_texts["LOW"])

    raw_log = str(telemetry.get("Latest_Syslog", "NORMAL_OPERATIONAL_STATE"))
    log_info = syslog_collector.parse_syslog(raw_log)

    shap_causes = shap_explainer.get_shap_causes(clean_telemetry, top_n=5)
    report = health_engine.build_health_report(clean_telemetry, probability, shap_causes, anom_res["anomaly_score"])

    result = {
        "success":              True,
        "device_id":            device_id,
        "prediction":           prediction,
        "probability":          probability,
        "risk":                 risk,
        "risk_color":           color,
        "status_text":          status_text,
        "model_used":           get_active_model_name(),
        "health_score":         report["health_score"],
        "risk_window":          report["risk_window"],
        "failure_type":         final_failure_type,
        "diagnosis_narrative":  diag_res["description"],
        "anomaly_score":        anom_res["anomaly_score"],
        "is_anomaly":           anom_res["is_anomaly"],
        "anomalous_features":   anom_res["anomalous_features"],
        "shap_causes":          shap_causes if shap_causes else report["top_causes"],
        "recommended_actions":  diag_res["recommended_actions"],
        "advisory":             diag_res["recommended_actions"],
        "syslog_summary":       log_info
    }

    if log_to_history:
        history_store.log_prediction(device_id, clean_telemetry, result)

    return result

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

@app.route('/api/devices', methods=['GET'])
def get_devices():
    try:
        devices = history_store.get_device_inventory()
        return jsonify({"success": True, "devices": devices})
    except Exception as e:
        return jsonify({"error": f"Devices fetch failed: {str(e)}"}), 500

@app.route('/api/topology', methods=['GET'])
def get_topology():
    """Returns interactive Network Topology node graph parsed from dataset v1 topology.csv."""
    top_csv = os.path.join(WORKSPACE_ROOT, 'data', 'netguard_noc_dataset_v1', 'topology.csv')
    dev_csv = os.path.join(WORKSPACE_ROOT, 'data', 'netguard_noc_dataset_v1', 'network_devices.csv')

    try:
        nodes = []
        links = []
        
        if os.path.exists(dev_csv):
            df_dev = pd.read_csv(dev_csv)
            sample_devs = df_dev.head(12).to_dict(orient="records")
            for dev in sample_devs:
                dev_id = str(dev.get("Device_ID"))
                nodes.append({
                    "id": dev_id,
                    "name": str(dev.get("Hostname", dev_id)),
                    "type": str(dev.get("Device_Type", "Switch")),
                    "vendor": str(dev.get("Vendor", "Cisco")),
                    "model": str(dev.get("Model", "ISR-4331")),
                    "location": str(dev.get("Location", "Chennai DC-1")),
                    "status": "HEALTHY",
                    "health": 92.0
                })
        else:
            nodes.append({"id": "DEV-0001", "name": "firewall-0001", "type": "ACCESS_SWITCH", "status": "HEALTHY", "health": 95.0})

        if os.path.exists(top_csv):
            df_top = pd.read_csv(top_csv)
            for _, row in df_top.head(20).iterrows():
                src = str(row.get("Source_Device_ID"))
                tgt = str(row.get("Target_Device_ID"))
                rel = str(row.get("Relationship"))
                if tgt and tgt != "nan":
                    links.append({"source": tgt, "target": src, "relationship": rel})

        return jsonify({"success": True, "nodes": nodes, "links": links})
    except Exception as e:
        return jsonify({"error": f"Topology load failed: {str(e)}"}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        alerts = history_store.get_active_alerts(limit=50)
        return jsonify({"success": True, "alerts": alerts})
    except Exception as e:
        return jsonify({"error": f"Alerts fetch failed: {str(e)}"}), 500

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert_route(alert_id):
    try:
        res = history_store.acknowledge_alert(alert_id)
        return jsonify({"success": res, "alert_id": alert_id, "status": "ACKNOWLEDGED"})
    except Exception as e:
        return jsonify({"error": f"Failed to acknowledge alert: {str(e)}"}), 500

@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
def resolve_alert_route(alert_id):
    try:
        res = history_store.resolve_alert(alert_id)
        return jsonify({"success": res, "alert_id": alert_id, "status": "RESOLVED"})
    except Exception as e:
        return jsonify({"error": f"Failed to resolve alert: {str(e)}"}), 500

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
    csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'netguard_noc_dataset_v1', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        return jsonify({"error": "Time-series dataset CSV not found."}), 404
    try:
        df = pd.read_csv(csv_path)
        dev_df = df[df["Device_ID"] == device_id].sort_values("Timestamp").tail(50)
        records = dev_df.to_dict(orient="records")
        return jsonify({"success": True, "device_id": device_id, "timeline": records})
    except Exception as e:
        return jsonify({"error": f"Timeline fetch failed: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def get_global_history():
    try:
        records = history_store.get_recent_global(limit=30)
        return jsonify({"success": True, "records": records})
    except Exception as e:
        return jsonify({"error": f"Global history fetch failed: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'netguard_noc_dataset_v1', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'network_devices_timeseries.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'network_devices.csv')

    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset CSV file not found."}), 404

    try:
        df = pd.read_csv(csv_path)
        latest_df = df.groupby("Device_ID").last().reset_index() if "Timestamp" in df.columns else df
            
        total_devices = len(latest_df)
        failed_count = int(latest_df['Failed'].sum())
        healthy_count = total_devices - failed_count
        health_rate = round((healthy_count / total_devices) * 100, 2)
        
        avg_cpu = float(latest_df['CPU_Usage'].mean())
        avg_mem = float(latest_df['Memory_Usage'].mean())
        avg_temp = float(latest_df['Temperature'].mean())
        avg_loss = float(latest_df['Packet_Loss'].mean())
        
        device_types = latest_df['Device_Type'].value_counts().to_dict()
        failure_types = latest_df['Failure_Type'].value_counts().to_dict() if 'Failure_Type' in latest_df.columns else {}

        return jsonify({
            "success": True,
            "dataset_name": "netguard_noc_dataset_v1",
            "total_devices": total_devices,
            "failed_count": failed_count,
            "healthy_count": healthy_count,
            "health_rate": health_rate,
            "avg_cpu": round(avg_cpu, 2),
            "avg_mem": round(avg_mem, 2),
            "avg_temp": round(avg_temp, 2),
            "avg_loss": round(avg_loss, 2),
            "routers_count": int(device_types.get('ROUTER', device_types.get('Router', 0))),
            "switches_count": int(device_types.get('ACCESS_SWITCH', device_types.get('Switch', 0))),
            "failure_types_breakdown": failure_types,
            "active_model": get_active_model_name()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load dataset statistics: {str(e)}"}), 500

@app.route('/api/plots/<filename>')
def serve_plot(filename):
    plots_dir = os.path.join(WORKSPACE_ROOT, 'outputs')
    return send_from_directory(plots_dir, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Launching NetGuard NOC web server on http://localhost:{port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
