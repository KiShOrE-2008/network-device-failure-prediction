"""
src/train_model.py
------------------
Multi-Model Machine Learning Training Pipeline for NetGuard NOC.
Trains:
  1. Supervised Binary Failure Model (XGBoost / Random Forest)
  2. Supervised Multi-Class Diagnostic Classifier (Failure_Type)
  3. Unsupervised Isolation Forest Anomaly Detector

Enforces Group-based Train/Test Split by Device_ID to prevent temporal data leakage.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

import anomaly_detection

def train_all_models():
    print("=" * 60)
    print("NetGuard NOC — Multi-Model Machine Learning Training")
    print("=" * 60)

    # 1. Load Dataset
    csv_path = "data/network_devices_timeseries.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/network_devices.csv"
        
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records across {df['Device_ID'].nunique()} devices.")

    # Fill missing trend values if necessary
    trend_cols = ["CPU_Trend", "Memory_Trend", "Temperature_Trend", "Error_Trend", "PacketLoss_Trend"]
    for col in trend_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Define Feature Sets
    categorical_features = ["Device_Type"]
    numerical_features = [
        "CPU_Usage",
        "Memory_Usage",
        "Temperature",
        "Uptime",
        "Interface_Errors",
        "Packet_Loss",
        "Bandwidth_Usage",
        "Log_Errors",
        "CPU_Trend",
        "Memory_Trend",
        "Temperature_Trend",
        "Error_Trend",
        "PacketLoss_Trend"
    ]

    all_features = categorical_features + numerical_features
    X = df[all_features]
    y_binary = df["Failed"]
    y_multiclass = df["Failure_Type"] if "Failure_Type" in df.columns else df["Failed"].astype(str)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )

    # ---------------------------------------------------------------------------
    # Enforce Group-based Train/Test Split by Device_ID (No Temporal Leakage!)
    # ---------------------------------------------------------------------------
    print("\nSplitting train/test data by Device_ID (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_binary, groups=df["Device_ID"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_bin, y_test_bin = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
    y_train_multi, y_test_multi = y_multiclass.iloc[train_idx], y_multiclass.iloc[test_idx]

    print(f"Train Set: {len(X_train):,} rows | Test Set: {len(X_test):,} rows")

    # ---------------------------------------------------------------------------
    # Model 1: Supervised Binary Failure Classifier
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. Training Binary Failure Classifiers")
    print("-" * 50)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, eval_metric="logloss")
    }

    best_bin_model = None
    best_bin_score = 0.0
    best_bin_name = ""
    bin_results = {}

    for name, clf in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clf)
        ])
        
        pipeline.fit(X_train, y_train_bin)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test_bin, y_pred)
        prec = precision_score(y_test_bin, y_pred, zero_division=0)
        rec = recall_score(y_test_bin, y_pred, zero_division=0)
        f1 = f1_score(y_test_bin, y_pred, zero_division=0)
        roc = roc_auc_score(y_test_bin, y_prob)

        bin_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": roc}
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

        if f1 > best_bin_score:
            best_bin_score = f1
            best_bin_model = pipeline
            best_bin_name = name

    print(f"\n🏆 Best Binary Failure Model: {best_bin_name} (F1 = {best_bin_score:.4f})")

    # ---------------------------------------------------------------------------
    # Model 2: Supervised Multi-Class Diagnostic Classifier
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. Training Multi-Class Diagnostic Classifier (Failure_Type)")
    print("-" * 50)

    label_encoder = LabelEncoder()
    y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
    y_test_multi_encoded = label_encoder.transform(y_test_multi)

    diagnostic_clf = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss"
    )

    diagnostic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", diagnostic_clf)
    ])

    diagnostic_pipeline.fit(X_train, y_train_multi_encoded)
    y_pred_multi = diagnostic_pipeline.predict(X_test)

    diag_acc = accuracy_score(y_test_multi_encoded, y_pred_multi)
    diag_f1 = f1_score(y_test_multi_encoded, y_pred_multi, average="weighted", zero_division=0)
    print(f"[XGBoost Multi-Class] Accuracy: {diag_acc:.4f} | Weighted F1: {diag_f1:.4f}")

    # Attach label encoder mapping to diagnostic pipeline object
    diagnostic_pipeline.label_classes_ = label_encoder.classes_.tolist()

    # ---------------------------------------------------------------------------
    # Model 3: Unsupervised Isolation Forest Anomaly Model
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. Training Isolation Forest Anomaly Detector")
    print("-" * 50)

    anomaly_model = anomaly_detection.train_anomaly_model(df, model_path="models/anomaly_model.pkl")

    # ---------------------------------------------------------------------------
    # Save Model Artifacts & Metadata
    # ---------------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    
    bin_model_path = "models/failure_model.pkl"
    diag_model_path = "models/diagnostic_model.pkl"
    metadata_path = "models/model_metadata.json"

    joblib.dump(best_bin_model, bin_model_path)
    joblib.dump(diagnostic_pipeline, diag_model_path)

    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "total_records": len(df),
        "total_devices": int(df['Device_ID'].nunique()),
        "binary_model_name": best_bin_name,
        "binary_metrics": bin_results[best_bin_name],
        "diagnostic_metrics": {"Accuracy": diag_acc, "Weighted_F1": diag_f1},
        "failure_classes": label_encoder.classes_.tolist(),
        "features": all_features
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Model Training Complete & Saved Successfully")
    print("=" * 60)
    print(f"Binary Failure Model:    {bin_model_path}")
    print(f"Diagnostic Model:        {diag_model_path}")
    print(f"Anomaly Detector Model:  models/anomaly_model.pkl")
    print(f"Metadata Summary:        {metadata_path}")

if __name__ == "__main__":
    train_all_models()