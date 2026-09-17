# 🌐 Network Device Failure Prediction & Diagnostic Maintenance System

> **A Complete Technical Overview, Mathematical Foundation, Model Benchmark, and Deployment Architecture**

---

## 🎯 1. Executive Summary & Project Purpose

Modern enterprise IT and telecommunication networks rely on thousands of interconnected routers and switches. An unpredicted device outage can cascade into network degradation, data loss, and significant financial costs. 

The **Network Device Failure Prediction System** is an end-to-end predictive maintenance solution built with Machine Learning and modern Web technologies. It analyzes multi-dimensional telemetry indicators—such as CPU load, thermal levels, buffer memory, interface error rates, and packet loss—to estimate real-time device health, calculate failure probabilities, and trigger preventive maintenance alerts before a catastrophic failure occurs.

### Key Objectives
1. **Early Anomaly & Failure Detection**: Predict high-risk device states prior to physical failure.
2. **Physics-Informed Synthetic Data**: Generate realistic 10,000-device telemetry datasets incorporating physical stress degradation factors and noise.
3. **Automated ML Benchmarking**: Compare linear, bagging, and boosting algorithms (Logistic Regression, Random Forest, XGBoost) using F1-score evaluation.
4. **Interactive Diagnostics Web Dashboard**: Provide network engineers with an intuitive web application featuring real-time risk gauges, instant parameter synthesis, scenario presets, and visual EDA analytics.

---

## 🏗️ 2. System Architecture & Component Design

The project is structured modularly to separate data synthesis, exploratory visualization, model preprocessing/training, inference execution, and web serving.

```text
network-device-failure-prediction/
│
├── app.py                      # Master pipeline orchestrator script (CLI & Web launcher)
├── PROJECT_DESCRIPTION.md      # Comprehensive technical documentation & project guide
├── README.md                   # Quick-start documentation & repo overview
├── requirements.txt            # Python dependencies (scikit-learn, xgboost, flask, etc.)
├── Poster.jpeg                 # Project poster graphic
│
├── data/                       # Data storage directory
│   └── network_devices.csv     # Generated synthetic telemetry dataset (10,000 records)
│
├── models/                     # Model artifact registry
│   └── failure_model.pkl       # Serialized production ColumnTransformer + XGBoost pipeline
│
├── outputs/                    # Output visual analytics generated during EDA
│   ├── cpu_vs_failure.png      # CPU usage vs failure distribution boxplot
│   ├── memory_vs_failure.png   # Memory usage vs failure distribution boxplot
│   ├── temperature_vs_failure.png # Thermal load vs failure boxplot
│   ├── failure_distribution.png   # Class distribution bar chart
│   └── correlation_heatmap.png    # Feature Pearson correlation matrix
│
└── src/                        # Core Python implementation & Web assets
    ├── generate_dataset.py     # Physics-informed synthetic dataset generator
    ├── eda.py                  # Exploratory data analysis & automated chart generator
    ├── preprocess.py           # Preprocessing utilities & feature definitions
    ├── train_model.py          # Preprocessing + multi-model training + evaluation pipeline
    ├── predict.py              # Interactive & non-interactive CLI inference utility
    ├── web_app.py              # Flask REST API backend server
    └── static/                 # Frontend Single Page Application (SPA)
        ├── index.html          # Web dashboard interface HTML5 markup
        ├── style.css           # Premium glassmorphic styling system
        └── app.js              # Real-time AJAX client & dynamic gauge renderer
```

---

## 🔬 3. Data Generation & Physics-Informed Modeling

In production environments, telemetry datasets are often proprietary or hard to export. This system includes a synthetic data engine ([generate_dataset.py](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/src/generate_dataset.py)) that synthesizes operational parameters for **10,000 network devices** (Routers and Switches).

### Statistical Distributions & Features
- **Device ID**: Sequential unique identifier (`DEV-00001` to `DEV-10000`).
- **Device Type**: Discrete distribution (`Router`, `Switch`).
- **CPU & Memory Usage**: Continuous uniform distribution ($10\% - 100\%$).
- **Temperature**: Thermal reading in Celsius uniform distribution ($25^\circ\text{C} - 90^\circ\text{C}$).
- **Uptime**: Operational longevity in days ($1 - 1000$ days).
- **Interface Errors**: Discrete Poisson distribution ($\lambda = 20$).
- **Packet Loss**: Percentage uniform distribution ($0\% - 10\%$).
- **Bandwidth Usage**: Active interface throughput utilization ($10\% - 100\%$).
- **Syslog Errors**: System error event counter Poisson distribution ($\lambda = 5$).

### Mathematical Failure Risk Equation

Rather than assigning labels at random, the target binary variable `Failed` ($0$ = Healthy, $1$ = Failure Imminent) is computed using a weighted linear combination of normalized operational metrics plus Gaussian noise:

$$S_{\text{failure}} = 0.25 \left(\frac{\text{CPU\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Memory\_Usage}}{100}\right) + 0.20 \left(\frac{\text{Temperature}}{90}\right) + 0.10 \left(\frac{\text{Interface\_Errors}}{100}\right) + 0.10 \left(\frac{\text{Packet\_Loss}}{10}\right) + 0.10 \left(\frac{\text{Bandwidth\_Usage}}{100}\right) + 0.05 \left(\frac{\text{Log\_Errors}}{30}\right) + \epsilon$$

Where:
- $\epsilon \sim \mathcal{N}(0, 0.05)$ represents hardware component variance and environmental noise.
- The failure condition is thresholded at $S_{\text{failure}} > 0.65$:

$$\text{Failed} = \begin{cases} 1 & \text{if } S_{\text{failure}} > 0.65 \\ 0 & \text{otherwise} \end{cases}$$

---

## 📊 4. Telemetry Data Dictionary

| Feature Name | Field Type | Data Range / Unit | Physics & Operational Context |
| :--- | :--- | :--- | :--- |
| `Device_ID` | String | `DEV-00001` to `DEV-10000` | Hardware host key identifier |
| `Device_Type` | Categorical | `["Router", "Switch"]` | Network hardware role profile |
| `CPU_Usage` | Float | $10.0\% - 100.0\%$ | Processor load; sustained high load causes process drops |
| `Memory_Usage` | Float | $20.0\% - 100.0\%$ | Buffer memory load; exhaustion leads to packet drop & crash |
| `Temperature` | Float | $25.0^\circ\text{C} - 90.0^\circ\text{C}$ | Thermal reading; overheating causes thermal throttling |
| `Uptime` | Float | $1.0 - 1000.0$ days | Days since boot; long uptime can expose memory leak issues |
| `Interface_Errors` | Integer | Counter ($\ge 0$) | Physical layer CRC / alignment packet frame error counter |
| `Packet_Loss` | Float | $0.0\% - 10.0\%$ | Fraction of dropped network packets across interfaces |
| `Bandwidth_Usage` | Float | $10.0\% - 100.0\%$ | Total interface throughput saturation level |
| `Log_Errors` | Integer | Counter ($\ge 0$) | System log error severity event count in buffer |
| **`Failed`** | Binary | `0` (Healthy) or `1` (Failed) | Target output variable to predict |

---

## ⚙️ 5. Preprocessing & Feature Engineering Pipeline

Preprocessing is implemented inside [train_model.py](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/src/train_model.py) using `scikit-learn` primitives encapsulated within a unified `Pipeline` to prevent target leakage:

1. **Categorical Features (`Device_Type`)**:
   - Processed via `OneHotEncoder(drop="first", sparse_output=False)` to produce binary indicators (`Device_Type_Switch`).
2. **Numerical Features**:
   - Transformed using `StandardScaler` to ensure zero mean ($\mu=0$) and unit variance ($\sigma^2=1$):
     $$z = \frac{x - \mu}{\sigma}$$
3. **Data Splitting**:
   - $80\%$ Training set ($8,000$ samples), $20\%$ Test set ($2,000$ samples).
   - Stratified train-test split (`stratify=y`) ensures identical class ratio preservation across sets.

---

## 🤖 6. Machine Learning Model Training & Evaluation

Three distinct machine learning models are trained and benchmarked against the test set:

### Models Evaluated
1. **Logistic Regression**: Baseline linear model (`max_iter=1000`).
2. **Random Forest Classifier**: Ensemble bagging classifier (`n_estimators=200`, `class_weight="balanced"`).
3. **XGBoost Classifier**: Gradient boosted decision tree (`n_estimators=200`, `max_depth=5`, `learning_rate=0.05`).

### Performance Metrics Benchmark

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $94.10\%$ | $86.54\%$ | $67.87\%$ | **$0.7608$** | $0.9765$ | Baseline |
| **Random Forest** | $96.50\%$ | $88.76\%$ | $85.20\%$ | **$0.8694$** | $0.9880$ | High Performance |
| **XGBoost** 🏆 | **$97.35\%$** | **$92.34\%$** | **$87.00\%$** | **$0.8959$** | **$0.9922$** | **Selected Model** |

### Primary Metric Rationale (F1-Score)
In network reliability engineering:
- **False Negatives (Missed Failures)**: Device fails silently without warning $\rightarrow$ costly network outage.
- **False Positives (False Alarms)**: High alert rate causes alarm fatigue and unnecessary manual technician dispatch.

The **F1-Score** ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$) provides the optimal balance. **XGBoost** achieved the highest F1-Score of **$0.8959$** and an ROC-AUC of **$0.9922$**, making it the champion model saved to `models/failure_model.pkl`.

---

## 💻 7. Interactive Diagnostics Web Interface

The web interface is hosted by Flask ([src/web_app.py](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/src/web_app.py)) and rendered by an interactive Single Page Application ([src/static/index.html](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/src/static/index.html)):

### Key Features
- **Real-Time Dynamic Assessment**: Drag telemetry sliders (CPU, Memory, Temp, Errors, etc.) to get instant debounced prediction calculations without page reloads.
- **SVG Circular Risk Gauge**: Animated indicator with color coding:
  - 🟢 **LOW RISK** ($0\% - 40\%$ failure probability)
  - 🟡 **MODERATE RISK** ($40\% - 70\%$ failure probability)
  - 🔴 **HIGH RISK** ($70\% - 100\%$ failure probability)
- **Preset Telemetry Profiles**: One-click quick load scenarios for:
  - *Nominal Baseline*
  - *Thermal Overheat*
  - *Network Congestion / Buffer Strain*
- **EDA Analytics Gallery**: Interactive modal displaying generated graphs ([cpu_vs_failure.png](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/outputs/cpu_vs_failure.png), [correlation_heatmap.png](file:///run/media/kishore/Data/SEM%20-%203/project/network-device-failure-prediction/outputs/correlation_heatmap.png), etc.).
- **Model Performance Cards**: Live view of trained classifier metrics.

---

## 🚀 8. Running the Application

### 1. Environment Setup
```bash
# Activate Python virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Execution Options

#### Option A: Launch the Diagnostics Web App
```bash
python app.py --web
# Server starts at http://localhost:5000
```

#### Option B: Run Full Pipeline via Master Script
```bash
python app.py
```

#### Option C: Execute Pipeline Components Sequentially
```bash
# Step 1: Generate synthetic telemetry data
python src/generate_dataset.py

# Step 2: Perform EDA & plot output charts
python src/eda.py

# Step 3: Preprocess data, train models, save optimal pipeline
python src/train_model.py

# Step 4: Run interactive CLI inference tool
python src/predict.py
```

---

## 🔗 9. REST API Reference

The Flask web application exposes the following JSON REST endpoints:

- `POST /api/predict`: Evaluates telemetry payload and returns predicted class, failure probability, risk label, gauge angle, and action message.
- `GET /api/model_info`: Returns trained model metadata, accuracy, precision, recall, and F1 scores.
- `GET /api/plots/<filename>`: Serves generated EDA visual charts.

---

## 🔮 10. Summary & Future Roadmap

This project demonstrates a production-grade predictive maintenance workflow for networking infrastructure. Future enhancements include:
1. **Prometheus / SNMP Integration**: Polling live telemetry streams directly from physical Cisco / Juniper devices.
2. **Time-Series Sequence Modeling**: Integrating LSTM / GRU networks for dynamic temporal failure prediction.
3. **Automated Incident Remediation**: Triggering Ansible playbooks or webhooks to reroute network traffic automatically when risk exceeds $75\%$.












