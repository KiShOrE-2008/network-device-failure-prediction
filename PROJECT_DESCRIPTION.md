# 🌐 NetGuard NOC — Fleet-Wide Predictive Failure Intelligence & Discovery

> **A Comprehensive Architectural Blueprint, Temporal ML Methodology, Multi-Mode Discovery System, and Deployment Guide**

---

## 🎯 1. Executive Summary & Project Purpose

Modern enterprise IT and telecommunication networks rely on thousands of interconnected routers and switches. An unpredicted device outage can cascade into network degradation, data loss, and significant financial costs.

The **NetGuard NOC System** is a fleet-wide predictive maintenance and network discovery platform built with Machine Learning and modern Web technologies. It analyzes multi-dimensional time-series telemetry—such as CPU load, thermal levels, buffer memory, interface error rates, and packet loss—to estimate real-time device health, calculate 12-hour impending failure probabilities without target leakage, and trigger preventive maintenance alerts before a catastrophic failure occurs.

### Key Objectives & 5 Hard Guarantees

1. **Zero Target & Future Leakage**: Deterministic chronological 60/20/20 per-device split (Train / Validation / Held-Out Test) with strict 12-step boundary purging (`PURGE_HORIZON = 12`).
2. **Past-Only Rolling Features**: All rolling averages, trends, and spikes compute exclusively over past observations (`shift(1)`).
3. **No Fake ML Probabilities**: Return explicit `prediction_available: false` and `model_status: "UNAVAILABLE"` when models are missing.
4. **Authentic Discovery Modes**: Clear separation between `SIMULATION`, `LAB` (socket probing), and `PRODUCTION` (SNMPv2c/v3, NETCONF, RESTCONF, Vendor REST API).
5. **Single Authoritative Schema Contract**: Centralized `FEATURE_COLUMNS`, `ANOMALY_FEATURE_COLUMNS`, `DIAGNOSTIC_FEATURE_COLUMNS`, and `ModelRegistry` schema validation (`model_registry.py`).


---

## 🏗️ 2. System Target Architecture

```text
                    ┌─────────────────────┐
                    │ Authorized Network  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Network Discovery   │
                    │ SIM / LAB / PROD    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Device Inventory    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Telemetry Pipeline  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               ↓
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
   Failure Prediction      Anomaly Detection    Diagnosis
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Fleet Intelligence  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Health + Risk       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Incident Engine     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ REST API            │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ NOC Dashboard       │
                    └──────────┬──────────┘
```

```text
network-device-failure-prediction/
│
├── app.py                      # Root delegator script for CLI & Web server execution
├── PROJECT_DESCRIPTION.md      # Comprehensive technical documentation & project guide
├── README.md                   # Quick-start documentation & repo overview
│
├── frontend/                   # Frontend Single Page Application (SPA)
│   ├── index.html              # Web dashboard interface HTML5 markup
│   ├── style.css               # Modern styling system & dashboard theme
│   ├── app.js                  # Real-time AJAX client & dynamic visual chart renderer
│   ├── app_additions.js        # Extended dashboard interactive modules
│   └── logo.png                # NetGuard NOC brand icon
│
└── backend/                    # Backend API, AI models, data & test suite
    ├── app.py                  # Master backend pipeline orchestrator script
    ├── requirements.txt        # Python dependencies
    │
    ├── data/                   # Data storage directory
    │   ├── network_devices_timeseries.csv # Benchmark time-series dataset
    │   └── network_noc.db      # Relational SQLite database
    │
    ├── models/                 # Model artifact registry
    │   ├── failure_model.pkl   # Supervised XGBoost binary forecaster
    │   ├── diagnostic_model.pkl# Multi-class root-cause classifier
    │   ├── anomaly_model.pkl   # Isolation Forest anomaly detector pipeline
    │   └── model_metadata.json # Auto-generated contract metadata
    │
    ├── tests/                  # Automated pytest test suite (58+ passing tests)
    │   ├── test_ml_integrity.py
    │   ├── test_integration.py
    │   ├── test_model_contract.py
    │   └── ...
    │
    └── src/                    # Core Python implementation
        ├── api/                # Modular REST API blueprints (fleet, devices, alerts, topology, discovery)
        ├── services/           # Service abstraction layer (inference, fleet, anomaly, incident)
        ├── intelligence/       # Multi-class root-cause diagnostic engine
        ├── feature_engineering.py # Authoritative feature schema & rolling transformations
        ├── target_engineering.py  # 12-hour lookahead target engine & purge horizon
        ├── train_model.py      # Chronological temporal training pipeline
        ├── fleet_predictor.py  # Vectorized fleet batch prediction engine
        ├── discovery.py        # Multi-mode network discovery engine & ProductionDiscovery adapters
        ├── model_registry.py   # Model schema contract validator
        ├── history_store.py    # Relational SQLite database & deduplicated incident store
        └── web_app.py          # Flask REST API server
```

---

## 🔬 3. Data Generation & Physics-Informed Modeling

In production environments, telemetry datasets are often proprietary or hard to export. This system includes a synthetic data engine (`generate_dataset.py`) that synthesizes operational parameters for **500 network devices** over time-series observations.

### Failure Probability Score Equation

$$S_{failure} = 0.25 \left(\frac{\text{CPU\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Memory\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Temperature}}{90}\right) + 0.10 \left(\frac{\text{Interface\_Errors}}{100}\right) + 0.10 \left(\frac{\text{Packet\_Loss}}{10}\right) + 0.10 \left(\frac{\text{Bandwidth\_Usage}}{100}\right) + 0.05 \left(\frac{\text{Log\_Errors}}{30}\right) + \epsilon$$

Where $\epsilon \sim \mathcal{N}(0, 0.05)$ represents environmental noise.

---

## 🚀 4. Running the Application

### 1. Environment Setup

```bash
# Activate Python virtual environment
source venv/bin/activate

# Install required packages
pip install -r backend/requirements.txt
```

### 2. Execution Options

#### Option A: Launch the Diagnostics Web App

```bash
python backend/app.py --web
# Server starts at http://localhost:5000
```

#### Option B: Run Full Pipeline via Master Script

```bash
python backend/app.py
```

#### Option C: Execute Pipeline Tests

```bash
venv/bin/pytest backend/tests
```

---

## 🔗 5. REST API Reference

The Flask web application exposes the following JSON REST endpoints:

- `GET /api/v1/health`: Returns system status and active model status.
- `GET /api/fleet/predictions`: Returns predictions for all fleet devices sorted by risk.
- `GET /api/fleet/stats`: Returns high-level NOC fleet statistics and health breakdown.
- `GET /api/fleet/health`: Returns overall network health score and operational status.
- `POST /api/fleet/predict`: Triggers batch inference and logs prediction history.
- `POST /api/predict`: Single-device evaluation payload response.
- `POST /api/discovery/scan`: Multi-mode network discovery scanner (`SIMULATION`, `LAB`, `PRODUCTION`).

---

## 🔮 6. Summary & Technical Limitations

1. **Synthetic Telemetry**: Benchmark dataset is synthetically generated for benchmark evaluation.
2. **Authorized Discovery**: Production discovery mode requires authenticated protocols (`SNMPv3`, `NETCONF`, `RESTCONF`).
3. **Anomaly vs Probability**: Unsupervised Anomaly Index measures baseline statistical deviation, distinct from failure probability.
