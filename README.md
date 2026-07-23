# Network Device Failure Prediction Pipeline

A comprehensive machine learning pipeline designed to predict failures in network devices (Routers and Switches) based on multi-dimensional telemetry, performance, and traffic quality metrics. The repository features synthetic data generation, exploratory visual analytics, comparative model training with evaluation metrics, and a predictive maintenance inference interface.

---

## 📖 Table of Contents
1. [Project Architecture](#project-architecture)
2. [Data Generation & Physics Simulation](#data-generation--physics-simulation)
3. [Telemetry Features & Data Dictionary](#telemetry-features--data-dictionary)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
6. [Model Architecture & Hyperparameters](#model-architecture--hyperparameters)
7. [Installation & Requirements](#installation--requirements)
8. [Usage Instructions](#usage-instructions)
9. [Sample Commands & CLI Output](#sample-commands--cli-output)
10. [Model Persistence & Registration](#model-persistence--registration)

---

## 🛠️ Project Architecture

```text
network-device-failure-prediction/
│
├── data/                       # Holds raw and generated CSV datasets
│   └── network_devices.csv     # Simulated telemetry dataset
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
│   └── predict.py              # CLI utility for inference on new device telemetry
│
├── requirements.txt            # System dependencies
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

- **Categorical Columns**: `Device_Type` is passed through for models that handle discrete features natively or via manual pipeline configurations.
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

### 2. Setup Guide
```bash
# Activate your virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

---

## 🚀 Usage Instructions

Execute the pipeline scripts in the following order:

```bash
# Step 1: Generate simulated dataset
python src/generate_dataset.py

# Step 2: Run Exploratory Data Analysis and generate charts
python src/eda.py

# Step 3: Run model training, comparison and save the best model
python src/train_model.py

# Step 4: Run test predictions on a sample payload
python src/predict.py
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

### 3. Inference / Prediction Output
```text
==================================================
NETWORK DEVICE FAILURE PREDICTION
==================================================
Device Type: Router
CPU Usage: 92.0%
Memory Usage: 94.0%
Temperature: 78.0°C
Interface Errors: 156
Packet Loss: 8.2%

Failure Probability: 95.34%
Risk Level: HIGH

⚠️ WARNING: Preventive maintenance recommended.
```

---

## 💾 Model Persistence & Registration

The serialized model is saved to `models/failure_model.pkl` as a unified `scikit-learn` `Pipeline` object containing:
1. `ColumnTransformer` (Scaling numeric values, leaving categoricals passthrough).
2. The optimized estimator (e.g., `XGBClassifier` or `RandomForestClassifier`).

You can load the model back in any Python process for production batch or API serving using `joblib`:

```python
import joblib
import pandas as pd

# Load saved pipeline
model_pipeline = joblib.load("models/failure_model.pkl")

# Predict on new data DataFrame
predictions = model_pipeline.predict(new_data_df)
probabilities = model_pipeline.predict_proba(new_data_df)[:, 1]
```
