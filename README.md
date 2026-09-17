# Network Device Failure Prediction Pipeline

A comprehensive machine learning pipeline designed to predict failures in network devices (Routers and Switches) based on multi-dimensional telemetry, performance, and traffic quality metrics. The repository features synthetic data generation, exploratory visual analytics, comparative model training, a predictive maintenance inference interface, and — as of v2 — an **Intelligence Hub** with real-time AI explainability, device health scoring, prediction history, and an interactive what-if simulator.

---

## 📖 Table of Contents
1. [Project Architecture](#project-architecture)
2. [Data Generation & Physics Simulation](#data-generation--physics-simulation)
3. [Telemetry Features & Data Dictionary](#telemetry-features--data-dictionary)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
6. [Model Architecture & Hyperparameters](#model-architecture--hyperparameters)
7. [Intelligence Hub — v2 Features](#intelligence-hub--v2-features)
8. [REST API Reference](#rest-api-reference)
9. [Installation & Requirements](#installation--requirements)
10. [Usage Instructions](#usage-instructions)
11. [Sample Commands & CLI Output](#sample-commands--cli-output)
12. [Model Persistence & Registration](#model-persistence--registration)

---

## 🛠️ Project Architecture

```text
network-device-failure-prediction/
│
├── data/                       # Holds raw and generated CSV datasets
│   ├── network_devices.csv     # Simulated telemetry dataset (10,000 records)
│   └── predictions.db          # SQLite prediction history (auto-created on first run)
│
├── models/                     # Registry for serialized joblib models
│   └── failure_model.pkl       # Saved optimal ColumnTransformer + Classifier pipeline
│
├── outputs/                    # Output visual analytics generated during EDA
│   ├── cpu_vs_failure.png
│   ├── memory_vs_failure.png
│   ├── temperature_vs_failure.png
│   ├── failure_distribution.png
│   └── correlation_heatmap.png
│
├── src/                        # Python codebase
│   ├── generate_dataset.py     # Generates synthetic network device dataset
│   ├── eda.py                  # Generates graphs and summary statistics
│   ├── preprocess.py           # Preprocessing utilities
│   ├── train_model.py          # Preprocessing + training + evaluation + saving pipeline
│   ├── predict.py              # CLI utility for inference on new device telemetry
│   ├── web_app.py              # Flask server backend (v2: enriched predict + new routes)
│   │
│   ├── health_engine.py        # [v2] Health score, risk window & cause-ranking engine
│   ├── shap_explainer.py       # [v2] SHAP TreeExplainer — AI feature attribution
│   ├── history_store.py        # [v2] SQLite-backed prediction history store
│   │
│   └── static/                 # Frontend SPA directory
│       ├── index.html          # Dashboard HTML UI (v2: Intelligence Hub tab + What-If)
│       ├── style.css           # Premium glassmorphic styling theme
│       ├── app.js              # Interactivity & AJAX client JavaScript
│       └── app_additions.js    # [v2] Health gauge, SHAP bars, history chart, what-if
│
├── requirements.txt            # System dependencies
├── app.py                      # Master pipeline orchestrator script (supports --web)
└── README.md                   # Complete pipeline documentation (this file)
```

---

## 🔬 Data Generation & Physics Simulation

Since production telemetry can contain sensitive topology information, the project generates realistic synthetic data (`src/generate_dataset.py`) for $10,000$ devices. Device failure labels are not set at random; instead, they are generated using a physics-informed risk function representing typical degradation factors in enterprise equipment:

### Failure Probability Score Equation

$$S_{failure} = 0.25 \left(\frac{\text{CPU\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Memory\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Temperature}}{90}\right) + 0.10 \left(\frac{\text{Interface\_Errors}}{100}\right) + 0.10 \left(\frac{\text{Packet\_Loss}}{10}\right) + 0.10 \left(\frac{\text{Bandwidth\_Usage}}{100}\right) + 0.05 \left(\frac{\text{Log\_Errors}}{30}\right) + \epsilon$$

Where:
- $\epsilon \sim \mathcal{N}(0, 0.05)$ represents stochastic environment/hardware noise.
- The failure threshold is set at $S_{failure} > 0.65$:
  $$\text{Failed} = \begin{cases} 1 & \text{if } S_{failure} > 0.65 \\ 0 & \text{otherwise} \end{cases}$$

> **Note:** The same weights are reused in `health_engine.py` to compute the real-time device health score (0–100), keeping the synthetic label formula and the live diagnostic engine conceptually consistent.

---

## 📊 Telemetry Features & Data Dictionary

| Feature Name | Type | Range / Distribution | Description |
| :--- | :--- | :--- | :--- |
| `Device_ID` | String | `DEV-00001` to `DEV-10000` | Primary key representing the network host |
| `Device_Type` | Categorical | `["Router", "Switch"]` | Device hardware profile |
| `CPU_Usage` | Float | $10.0\%$ to $100.0\%$ (Uniform) | Instantaneous CPU core utilization |
| `Memory_Usage` | Float | $20.0\%$ to $100.0\%$ (Uniform) | System RAM utilization percentage |
| `Temperature` | Float | $25.0^\circ\text{C}$ to $90.0^\circ\text{C}$ (Uniform) | Internal device temperature |
| `Uptime` | Float | $1.0$ to $1000.0$ days (Uniform) | Days since last restart/reboot |
| `Interface_Errors` | Integer | $\lambda = 20$ (Poisson) | Counter for interface CRC/alignment packet errors |
| `Packet_Loss` | Float | $0.0\%$ to $10.0\%$ (Uniform) | Packet drop rate percentage |
| `Bandwidth_Usage` | Float | $10.0\%$ to $100.0\%$ (Uniform) | Active port/interface speed consumption percentage |
| `Log_Errors` | Integer | $\lambda = 5$ (Poisson) | Error entries in the syslog buffer |
| **Failed** | Binary | `[0 (Healthy), 1 (Failed)]` | Target variable to predict |

---

## 📈 Exploratory Data Analysis (EDA)

The `src/eda.py` script reads the dataset and automatically performs univariate, bivariate, and multivariate analysis:

1. **Univariate Distribution Check**: Computes missing values and descriptive summaries.
2. **Failure Balance Visualizer**: Generates `outputs/failure_distribution.png` to analyze label distribution skewness.
3. **Degradation Analysis Plots**:
   - `outputs/cpu_vs_failure.png`
   - `outputs/memory_vs_failure.png`
   - `outputs/temperature_vs_failure.png`
   These box plots display how failure labels shift depending on operational thresholds.
4. **Correlation Heatmap (`outputs/correlation_heatmap.png`)**: Computes Pearson correlation matrices on numeric features to look for multicollinearity and target relationship strength.

---

## ⚙️ Preprocessing & Feature Engineering

Features must undergo transformations before feeding into linear or ensemble models:

- **Categorical Columns**: `Device_Type` is encoded using a `OneHotEncoder(drop="first", sparse_output=False)` to safely convert discrete classes (`Router`/`Switch`) into numeric values for downstream classifiers.
- **Numerical Columns**: Scaled using a `StandardScaler` to bring variance and mean to a uniform scale ($\mu=0, \sigma^2=1$):
  $$z = \frac{x - \mu}{\sigma}$$
- **Composition**: Handled elegantly via `sklearn.compose.ColumnTransformer`, ensuring **no target leakage** occurs during train-test splitting.

---

## 🤖 Model Architecture & Hyperparameters

Three different algorithms are trained and evaluated in parallel under `src/train_model.py`:

### 1. Logistic Regression
- **Parameters**: `max_iter=1000`
- **Utility**: Serves as a fast, interpretable linear baseline.

### 2. Random Forest Classifier
- **Parameters**: `n_estimators=200`, `random_state=42`, `class_weight="balanced"`
- **Utility**: Bagging ensemble, robust to outliers and feature interactions. Weighted classes address sample imbalances.

### 3. XGBoost Classifier
- **Parameters**: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `random_state=42`, `eval_metric="logloss"`
- **Utility**: Highly optimized gradient boosted trees. Excels at high-dimensional tabular datasets.

### Model Selection Metric

Models are scored on a hold-out test set ($20\%$). The final model selection is determined by the **F1-Score**, which ensures high harmonic mean of precision and recall, critical in predictive maintenance to avoid missing real failures (false negatives) while preventing excessive false alarms (false positives):

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 🧠 Intelligence Hub — v2 Features

Version 2 adds four intelligence features on top of the existing ML pipeline. They are implemented as separate, independently testable modules and are **non-destructive** — all existing routes and frontend behaviour are preserved.

### 1. Device Health Score (`src/health_engine.py`)

A real-time **0–100 health score** computed from the same weighted metric formula used to generate training labels. No model is needed — it is a pure function of the incoming telemetry.

| Score Range | Status |
|---|---|
| 70 – 100 | 🟢 HEALTHY |
| 40 – 69 | 🟡 DEGRADED |
| 0 – 39 | 🔴 CRITICAL |

Each prediction response now also includes an **estimated failure window** (e.g. *"High risk — immediate maintenance required"*) mapped from the model's probability output via heuristic bucketing.

### 2. SHAP AI Explainability (`src/shap_explainer.py`)

Uses `shap.TreeExplainer` — the exact, fast algorithm for XGBoost — to attribute the failure probability to individual telemetry features. Results are displayed as an animated bar chart in the **Intelligence Hub** tab.

- Each bar shows the **SHAP contribution** for that feature on the current telemetry reading.
- Red bars (↑) increase failure risk; green bars (↓) decrease it.
- If SHAP throws for any reason (version mismatch, model not loaded), the dashboard falls back to `health_engine.main_causes()` — a simpler weighted-badness ranking — so the UI **never breaks**.

### 3. Prediction History (`src/history_store.py`)

Every `/api/predict` call is automatically logged to a local **SQLite database** (`data/predictions.db`).

- The **Intelligence Hub → Prediction History** card displays a sparkline of the last 30 failure probabilities, with colour-coded dots per risk level, plus a tabular summary of the 10 most recent entries.
- You can look up history for any **Device ID** (e.g. `DEV-00042`) via the search field.
- `/api/whatif` calls are **intentionally not logged**, so slider-dragging in What-If mode never pollutes history.

### 4. What-If Simulator

A **What-If Simulator** button in the Diagnostics tab activates a live comparison mode:

- All existing sliders still drive the main prediction gauge (via `/api/predict`).
- With What-If mode **on**, slider changes additionally call `/api/whatif` (no logging) and display a delta panel showing:
  - Current vs baseline **failure probability** (with Δ coloured red/green)
  - Current vs baseline **health score** (with Δ)
  - Updated **risk level** and **estimated failure window**

### New API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/predict` | POST | Existing endpoint — now also returns `health_score`, `risk_window`, `shap_causes`, `recommended_actions` |
| `POST /api/whatif` | POST | Identical inference but does **not** log to history |
| `GET /api/history/<device_id>` | GET | Returns last 50 logged predictions for the device |
| `GET /api/history` | GET | Returns last 20 predictions across all devices |

---

## 🔌 REST API Reference

### `POST /api/predict`

**Request body (JSON):**
```json
{
  "device_id":        "DEV-00042",
  "Device_Type":      "Router",
  "CPU_Usage":        94.0,
  "Memory_Usage":     70.0,
  "Temperature":      81.0,
  "Uptime":           100.0,
  "Interface_Errors": 27,
  "Packet_Loss":      8.4,
  "Bandwidth_Usage":  60.0,
  "Log_Errors":       12
}
```

**Response (JSON) — v2 enriched:**
```json
{
  "success":             true,
  "prediction":          1,
  "probability":         0.82,
  "risk":                "HIGH",
  "risk_color":          "#ff1744",
  "status_text":         "CRITICAL WARNING: ...",
  "advisory":            ["CPU usage is critical. ...", "..."],
  "model_used":          "XGBoost Classifier",
  "health_score":        25.4,
  "risk_window":         "High risk — immediate maintenance required",
  "shap_causes":         [{"feature": "CPU Usage", "contribution": 0.2269, "direction": "increases_risk"}, "..."],
  "recommended_actions": ["Redistribute process load or upgrade CPU capacity.", "..."]
}
```

### `GET /api/history/<device_id>`
Returns `{"success": true, "device_id": "DEV-00042", "records": [...]}` with the last 50 logged predictions for that device.

### `GET /api/stats`
Dataset-level summary statistics (total devices, failure rate, average metrics, active model name).

### `GET /api/plots/<filename>`
Serves EDA plot images from `outputs/`.

---

## 📥 Installation & Requirements

### 1. Requirements
Ensure you are using Python 3.8+ with a virtual environment. The required libraries are:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `scikit-learn`
- `xgboost`
- `flask` (required for hosting the Web Interface dashboard)
- `shap` (required for AI explainability — v2)

### 2. Setup Guide
```bash
# Activate your virtual environment
source venv/bin/activate

# Install the required packages (includes shap as of v2)
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

### Run the Web Diagnostics Dashboard
You can host and run the interactive web interface locally using either:

```bash
# Option A: Start the web server using the master orchestrator
python app.py --web

# Option B: Run the web app script directly
python src/web_app.py
```
Once started, open your web browser and navigate to: **http://localhost:5000**.

#### Web Dashboard Features
- **Real-Time Telemetry Diagnosis**: Custom sliders with instant predictions (debounced) and an animated SVG risk gauge (Green/Amber/Red indicators).
- **Quick-presets**: Instantly populate nominal configurations, thermal failures, or congestion scenarios.
- **Reset to Defaults**: Reset all parameter inputs back to default standard values in one click.
- **Exploratory Analytics Gallery**: View and expand the EDA plots generated by the model.
- **Model Performance Registry**: Side-by-side comparison of Logistic Regression, Random Forest, and XGBoost classifiers.
- **🆕 Intelligence Hub tab**:
  - Circular health score gauge (0–100, colour-coded)
  - Estimated failure window label
  - Animated SHAP bar chart (top-5 feature attributions)
  - Prediction history sparkline + per-device lookup table
  - Recommended maintenance actions list
- **🆕 What-If Simulator**: Toggle in the Diagnostics tab to compare delta probability & health score against the baseline in real time — without logging to history.

---

### Run the Pipeline via CLI
You can also execute the batch pipeline using the orchestrator:

```bash
# Run the pipeline sequentially (interactive prediction step at the end)
python app.py

# Run the pipeline sequentially (non-interactive prediction step at the end)
python app.py --non-interactive
```

Alternatively, run individual scripts step-by-step:

```bash
# Step 1: Generate simulated dataset
python src/generate_dataset.py

# Step 2: Run Exploratory Data Analysis and generate charts
python src/eda.py

# Step 3: Run model training, comparison and save the best model
python src/train_model.py

# Step 4: Run predictions interactively (prompts for telemetry inputs, press Enter for defaults)
python src/predict.py

# Alternatively, run predictions non-interactively using the default sample payload
python src/predict.py --non-interactive
```

---

### Sanity-Check the v2 Modules

```bash
# Health engine (no model required)
python -c "
import sys; sys.path.insert(0, 'src')
from health_engine import build_health_report
print(build_health_report({'CPU_Usage':94,'Memory_Usage':70,'Temperature':81,'Interface_Errors':27,'Packet_Loss':8.4,'Bandwidth_Usage':60,'Log_Errors':12}, 0.82))
"

# History store (creates data/predictions.db if absent)
python -c "import sys; sys.path.insert(0, 'src'); import history_store; history_store.init_db(); print('ok')"
```

---

## 🖥️ Sample Commands & CLI Output

### 1. Dataset Generation Output
```text
==================================================
Network Device Dataset Generated
==================================================
Total Records: 10000

Dataset Columns:
['Device_ID', 'Device_Type', 'CPU_Usage', 'Memory_Usage', 'Temperature', 'Uptime', 'Interface_Errors', 'Packet_Loss', 'Bandwidth_Usage', 'Log_Errors', 'Failed']

Failure Distribution:
Failed
0    8615
1    1385
Name: count, dtype: int64

Dataset saved to: data/network_devices.csv
```

### 2. Model Training Output
```text
Dataset loaded successfully.

==================================================
Training Logistic Regression
Accuracy: 0.9410
Precision: 0.8654
Recall: 0.6787
F1 Score: 0.7608
ROC-AUC: 0.9765

==================================================
Training Random Forest
Accuracy: 0.9650
Precision: 0.8876
Recall: 0.8520
F1 Score: 0.8694
ROC-AUC: 0.9880

==================================================
Training XGBoost
Accuracy: 0.9735
Precision: 0.9234
Recall: 0.8700
F1 Score: 0.8959
ROC-AUC: 0.9922

==================================================
MODEL COMPARISON
==================================================
Logistic Regression
Accuracy: 0.9410
F1 Score: 0.7608

Random Forest
Accuracy: 0.9650
F1 Score: 0.8694

XGBoost
Accuracy: 0.9735
F1 Score: 0.8959

==================================================
Best Model: XGBoost
Best F1 Score: 0.8959
Model saved to: models/failure_model.pkl
```

### 3. Inference / Prediction Output (Interactive Mode)
```text
==================================================
NETWORK DEVICE FAILURE PREDICTION - INTERACTIVE MODE
==================================================
Please enter telemetry values or press [Enter] to use defaults.
--------------------------------------------------
Device Type (Router/Switch) [Default: Router]: 
CPU Usage (%) (0-100) [Default: 92.0]: 
Memory Usage (%) (0-100) [Default: 94.0]: 
Temperature (°C) (0-150) [Default: 78.0]: 
Uptime (days) (>=0) [Default: 20.0]: 
Interface Errors (count >=0) [Default: 156]: 
Packet Loss (%) (0-100) [Default: 8.2]: 
Bandwidth Usage (%) (0-100) [Default: 95.0]: 
Log Errors (count >=0) [Default: 20]: 

==================================================
PREDICTION RESULT
==================================================
Device Type:        Router
CPU Usage:          92.0%
Memory Usage:       94.0%
Temperature:        78.0°C
Uptime:             20.0 days
Interface Errors:   156
Packet Loss:        8.2%
Bandwidth Usage:    95.0%
Log Errors:         20
--------------------------------------------------
Failure Probability: 84.00%
Risk Level:          HIGH
--------------------------------------------------
⚠️ WARNING: Preventive maintenance recommended.
==================================================
```

---

## 💾 Model Persistence & Registration

The serialized model is saved to `models/failure_model.pkl` as a unified `scikit-learn` `Pipeline` object containing:
1. `ColumnTransformer` step named `"preprocessor"` (scaling numeric values, encoding categoricals).
2. The optimized estimator step named `"model"` (e.g., `XGBClassifier`).

You can load the model back in any Python process for production batch or API serving using `joblib`:

```python
import joblib
import pandas as pd

# Load saved pipeline
model_pipeline = joblib.load("models/failure_model.pkl")

# Predict on new data DataFrame
predictions   = model_pipeline.predict(new_data_df)
probabilities = model_pipeline.predict_proba(new_data_df)[:, 1]
```

The SHAP explainer (`src/shap_explainer.py`) also loads this pipeline automatically on import and uses `pipeline.named_steps["preprocessor"]` and `pipeline.named_steps["model"]` to compute attributions.
