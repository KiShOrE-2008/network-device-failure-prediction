# Network Device Failure Prediction

A machine learning pipeline designed to predict failures in network devices (Routers and Switches) based on telemetry and performance metrics. The project generates synthetic device data, performs Exploratory Data Analysis (EDA), trains multiple machine learning classifiers, and evaluates them to select and save the best performing model for predictive maintenance tasks.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Telemetry Features](#telemetry-features)
4. [Installation & Setup](#installation--setup)
5. [Pipeline Execution](#pipeline-execution)
   - [1. Data Generation](#1-data-generation)
   - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
   - [3. Model Training & Selection](#3-model-training--selection)
   - [4. Single Device Inference](#4-single-device-inference)
6. [Models Evaluated](#models-evaluated)
7. [License](#license)

---

## Project Overview

This repository implements an end-to-end Machine Learning pipeline to classify network device health. By analyzing system metrics (such as CPU, Memory, and Temperature) alongside traffic quality characteristics (Packet Loss, Bandwidth, and Interface Errors), it predicts whether a device is at risk of imminent failure.

The pipeline comprises:
- **Synthetic Data Generation**: Simulates realistic telemetry attributes with configurable noise.
- **EDA & Visualization**: Automatically generates distribution, correlation, and feature-relationship plots.
- **Model Comparison**: Trains Logistic Regression, Random Forest, and XGBoost classifiers.
- **Model Registry**: Persists the best pipeline (preprocessor + estimator) using F1-score metric.
- **Inference Script**: Provides a lightweight CLI interface to predict failure risk on live inputs.

---

## Folder Structure

```text
├── data/                       # Contains generated CSV dataset(s)
├── models/                     # Registry for serialization of trained models (.pkl)
├── outputs/                    # Output visualizations and graphs from EDA
├── src/                        # Source python scripts
│   ├── eda.py                  # Exploratory Data Analysis & Visualization
│   ├── generate_dataset.py    # Synthetic telemetry generator
│   ├── predict.py              # Risk prediction script (Inference)
│   ├── preprocess.py           # Preprocessing helpers
│   └── train_model.py          # Training, evaluation & model persistence pipeline
├── requirements.txt            # Project python dependencies
└── README.md                   # Project documentation (this file)
```

---

## Telemetry Features

The synthetic dataset mimics typical SNMP and NetFlow parameters collected from enterprise routers and switches:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `Device_ID` | Categorical | Unique identifier for each network device |
| `Device_Type` | Categorical | Device type designation (`Router` or `Switch`) |
| `CPU_Usage` | Numerical | Percentage of CPU utilization (10% - 100%) |
| `Memory_Usage` | Numerical | Percentage of Memory usage (20% - 100%) |
| `Temperature` | Numerical | System temperature in Celsius (25°C - 90°C) |
| `Uptime` | Numerical | Elapsed time since last reboot in days (1 - 1000 days) |
| `Interface_Errors` | Numerical | Logged input/output interface errors (Poisson distributed) |
| `Packet_Loss` | Numerical | Packet drop rate percentage (0% - 10%) |
| `Bandwidth_Usage` | Numerical | Interface bandwidth usage percentage (10% - 100%) |
| `Log_Errors` | Numerical | Error lines logged in syslog buffer |
| `Failed` | Label (Binary) | Target failure label (`0` for Healthy, `1` for Failed) |

---

## Installation & Setup

1. **Clone or locate the repository directory**:
   ```bash
   cd network-device-failure-prediction
   ```

2. **Set up a Virtual Environment** (optional but highly recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Execution

To run the full end-to-end pipeline, execute the scripts inside the `src/` directory in the following sequence:

### 1. Data Generation
Generates a raw dataset of 10,000 device records under `data/network_devices.csv`:
```bash
python src/generate_dataset.py
```

### 2. Exploratory Data Analysis
Reads the dataset, generates statistical summaries, and exports key visualization charts into the `outputs/` folder:
```bash
python src/eda.py
```
*Generated plots:*
- `outputs/failure_distribution.png`: Count of Healthy vs Failed devices.
- `outputs/cpu_vs_failure.png`, `outputs/memory_vs_failure.png`, `outputs/temperature_vs_failure.png`: Feature distributions grouped by target class.
- `outputs/correlation_heatmap.png`: Spearman/Pearson correlation coefficients among numeric parameters.

### 3. Model Training & Selection
Preprocesses features (Standard Scaling + Pass-through Categorical encoding), splits data into train/test (80/20 stratified split), evaluates candidate classifiers, and exports the best model into `models/failure_model.pkl`:
```bash
python src/train_model.py
```

### 4. Single Device Inference
Load the optimized estimator and run prediction metrics against hypothetical live telemetry payload:
```bash
python src/predict.py
```

---

## Models Evaluated

The model training step performs comparative analysis across the following frameworks:
1. **Logistic Regression**: Linear baseline with L2 regularization.
2. **Random Forest Classifier**: Non-linear ensemble model using bagging and balanced class weights.
3. **XGBoost Classifier**: Gradient boosted decision trees optimized for speed and accuracy.

Model selection selects the pipeline that scores the highest **F1-Score** on the hold-out validation set to ensure a balanced handle on both precision and recall.
