import os
import sys
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

# Resolve paths relative to the file location to prevent working directory issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='')

# --------------------------------
# Load model
# --------------------------------
MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'models', 'failure_model.pkl')
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Loaded machine learning pipeline from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")
        print("Backend starting in degraded mode. Please run python src/train_model.py to train.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")

# Helper to get active model name
def get_active_model_name():
    if model is None:
        return "None (Model not trained)"
    try:
        # Check if it's a pipeline and inspect the classifier step
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            clf_class = model.named_steps['model'].__class__.__name__
            if "XGB" in clf_class:
                return "XGBoost Classifier"
            elif "RandomForest" in clf_class:
                return "Random Forest Classifier"
            elif "LogisticRegression" in clf_class:
                return "Logistic Regression"
            return clf_class
        return model.__class__.__name__
    except:
        return "Custom Pipeline"

# --------------------------------
# Routes
# --------------------------------

@app.route('/')
def index():
    """Serves the main SPA page."""
    return app.send_static_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Receives telemetry details and returns failure probability."""
    if model is None:
        return jsonify({
            "error": "ML model is not loaded. Please train the model using src/train_model.py."
        }), 503

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract features and convert types
        device_type = str(data.get("Device_Type", "Router")).strip()
        # Standardize casing to match training
        device_type = "Router" if device_type.lower() == "router" else "Switch"

        cpu_usage = float(data.get("CPU_Usage", 0.0))
        memory_usage = float(data.get("Memory_Usage", 0.0))
        temperature = float(data.get("Temperature", 0.0))
        uptime = float(data.get("Uptime", 0.0))
        interface_errors = int(data.get("Interface_Errors", 0))
        packet_loss = float(data.get("Packet_Loss", 0.0))
        bandwidth_usage = float(data.get("Bandwidth_Usage", 0.0))
        log_errors = int(data.get("Log_Errors", 0))

        # Create single-record DataFrame
        features_df = pd.DataFrame([{
            "Device_Type": device_type,
            "CPU_Usage": cpu_usage,
            "Memory_Usage": memory_usage,
            "Temperature": temperature,
            "Uptime": uptime,
            "Interface_Errors": interface_errors,
            "Packet_Loss": packet_loss,
            "Bandwidth_Usage": bandwidth_usage,
            "Log_Errors": log_errors
        }])

        # Perform inference
        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])

        # Risk Classification
        if probability < 0.30:
            risk = "LOW"
            color = "#00e676"  # Bright Neon Green
            status_text = "Device condition is currently healthy and operating within acceptable parameters."
        elif probability < 0.70:
            risk = "MEDIUM"
            color = "#ffb300"  # Amber
            status_text = "Moderate risk detected. Recommend close monitoring and secondary diagnostics."
        else:
            risk = "HIGH"
            color = "#ff1744"  # Neon Red
            status_text = "CRITICAL WARNING: High failure probability. Preventive maintenance or immediate reboot/failover is highly recommended."

        # Diagnostics advisory details
        advisory = []
        if cpu_usage > 85.0:
            advisory.append("CPU usage is critical. Consider load shedding, routing optimizations, or scaling hardware.")
        if memory_usage > 90.0:
            advisory.append("System memory is nearly exhausted. Check for memory leaks or rogue system processes.")
        if temperature > 75.0:
            advisory.append("Internal temperature is high. Clean chassis vents, check cooling fan performance, or reduce room ambient heat.")
        if interface_errors > 100:
            advisory.append("High count of interface CRC errors. Inspect network cables and SFP optics for physical defects.")
        if packet_loss > 3.0:
            advisory.append("Severe packet loss detected. Investigate buffer bloat, link congestion, or duplex mismatches.")
        if log_errors > 15:
            advisory.append("Excessive system log errors. Check console buffers for underlying hardware failures or authorization issues.")
        if uptime > 365:
            advisory.append("Device uptime exceeds 1 year. Schedule a preventative maintenance reboot to refresh buffers and system memory.")

        if not advisory:
            advisory.append("No specific hardware anomalies detected. All telemetry lines are within standard parameters.")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "risk": risk,
            "risk_color": color,
            "status_text": status_text,
            "advisory": advisory,
            "model_used": get_active_model_name()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to perform prediction: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Computes summary statistics from network_devices.csv dataset."""
    csv_path = os.path.join(WORKSPACE_ROOT, 'data', 'network_devices.csv')
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Dataset CSV file not found. Please run the data generation script."
        }), 404

    try:
        df = pd.read_csv(csv_path)
        
        # Calculate summary values
        total_devices = len(df)
        failed_count = int(df['Failed'].sum())
        healthy_count = total_devices - failed_count
        health_rate = (healthy_count / total_devices) * 100
        
        # Means
        avg_cpu = float(df['CPU_Usage'].mean())
        avg_mem = float(df['Memory_Usage'].mean())
        avg_temp = float(df['Temperature'].mean())
        avg_loss = float(df['Packet_Loss'].mean())
        
        # Device Breakdown
        device_types = df['Device_Type'].value_counts().to_dict()
        routers_count = int(device_types.get('Router', 0))
        switches_count = int(device_types.get('Switch', 0))

        return jsonify({
            "success": True,
            "total_devices": total_devices,
            "failed_count": failed_count,
            "healthy_count": healthy_count,
            "health_rate": round(health_rate, 2),
            "avg_cpu": round(avg_cpu, 2),
            "avg_mem": round(avg_mem, 2),
            "avg_temp": round(avg_temp, 2),
            "avg_loss": round(avg_loss, 2),
            "routers_count": routers_count,
            "switches_count": switches_count,
            "active_model": get_active_model_name()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load dataset statistics: {str(e)}"}), 500

@app.route('/api/plots/<filename>')
def serve_plot(filename):
    """Serves generated EDA plots from the outputs directory."""
    plots_dir = os.path.join(WORKSPACE_ROOT, 'outputs')
    return send_from_directory(plots_dir, filename)

if __name__ == '__main__':
    # Run the development server
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Launching NetGuard NOC web server on http://localhost:{port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
