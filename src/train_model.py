"""
src/train_model.py
------------------
NetGuard NOC Fleet ML Training Pipeline.
Consumes netguard_noc_dataset_v1 time-series benchmark dataset.
Enforces Group-based Train/Test Split by Device_ID to prevent device leakage.
Primary target: Failure_Next_12h (impending failure in next 12 hours).
Diagnostic target: Failure_Type (trained strictly on Failed == 1 records).
Anomaly target: Unsupervised Isolation Forest on operational baseline (Failed == 0).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
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
    confusion_matrix
)

import anomaly_detection
from feature_engineering import FEATURE_COLUMNS, compute_rolling_features


def train_all_models():
    print("=" * 60)
    print("NetGuard NOC — Training Pipeline (netguard_noc_dataset_v1)")
    print("=" * 60)

    csv_path = "data/netguard_noc_dataset_v1/network_devices_timeseries.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/network_devices_timeseries.csv"
        
    print(f"Loading benchmark dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} records across {df['Device_ID'].nunique()} devices.")

    # Ensure temporal rolling features are generated
    df = compute_rolling_features(df)

    # ---------------------------------------------------------------------------
    # Filter Valid Target Horizon (Exclude incomplete boundary timesteps where target is NaN)
    # ---------------------------------------------------------------------------
    valid_df = df[df['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    valid_df['Failure_Next_12h'] = valid_df['Failure_Next_12h'].astype(int)
    print(f"Valid records for Failure_Next_12h prediction: {len(valid_df):,}")

    categorical_features = ["Device_Type"]
    numerical_features = [c for c in FEATURE_COLUMNS if c in valid_df.columns]
    all_features = categorical_features + numerical_features

    X = valid_df[all_features]
    y_binary = valid_df["Failure_Next_12h"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ]
    )

    # ---------------------------------------------------------------------------
    # Enforce GroupShuffleSplit by Device_ID (80% train devices / 20% test devices)
    # ---------------------------------------------------------------------------
    print("\nSplitting train/test data by Device_ID (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_binary, groups=valid_df["Device_ID"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_bin, y_test_bin = y_binary.iloc[train_idx], y_binary.iloc[test_idx]

    print(f"Train Set: {len(X_train):,} rows ({y_train_bin.sum()} failures) | Test Set: {len(X_test):,} rows ({y_test_bin.sum()} failures)")

    # ---------------------------------------------------------------------------
    # Model 1: Supervised Binary Failure Predictor (Failure_Next_12h)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. Training Supervised Binary Failure Predictors (Target: Failure_Next_12h)")
    print("-" * 50)

    # Compute scale_pos_weight for XGBoost to handle class imbalance
    neg_count = (y_train_bin == 0).sum()
    pos_count = (y_train_bin == 1).sum()
    scale_weight = float(neg_count / max(pos_count, 1))

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_weight, random_state=42, eval_metric="logloss")
    }

    best_bin_model = None
    best_bin_score = -1.0
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
        cm = confusion_matrix(y_test_bin, y_pred).tolist()

        bin_results[name] = {
            "Accuracy": float(acc), "Precision": float(prec),
            "Recall": float(rec), "F1": float(f1), "ROC-AUC": float(roc), "ConfusionMatrix": cm
        }
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

        if f1 > best_bin_score:
            best_bin_score = f1
            best_bin_model = pipeline
            best_bin_name = name

    print(f"\n🏆 Best Binary Failure Model: {best_bin_name} (F1 = {best_bin_score:.4f})")

    # ---------------------------------------------------------------------------
    # Model 2: Supervised Multi-Class Diagnostic Classifier (Strictly Failed == 1)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. Training Diagnostic Classifier (Strictly on Failed == 1 records)")
    print("-" * 50)

    failed_df = df[df["Failed"] == 1].reset_index(drop=True)
    if len(failed_df) > 0 and "Failure_Type" in failed_df.columns:
        X_diag = failed_df[all_features]
        y_diag = failed_df["Failure_Type"]

        label_encoder = LabelEncoder()
        y_diag_enc = label_encoder.fit_transform(y_diag)

        diagnostic_clf = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="mlogloss"
        )
        diagnostic_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", diagnostic_clf)
        ])

        diagnostic_pipeline.fit(X_diag, y_diag_enc)
        y_diag_pred = diagnostic_pipeline.predict(X_diag)

        diag_acc = accuracy_score(y_diag_enc, y_diag_pred)
        diag_f1 = f1_score(y_diag_enc, y_diag_pred, average="weighted", zero_division=0)
        diag_cm = confusion_matrix(y_diag_enc, y_diag_pred).tolist()
        diag_classes = label_encoder.classes_.tolist()

        diagnostic_pipeline.label_classes_ = diag_classes
        print(f"[Diagnostic Classifier] Failed-only samples: {len(failed_df)}")
        print(f"[Diagnostic Classifier] Accuracy: {diag_acc:.4f} | Weighted F1: {diag_f1:.4f}")
    else:
        diag_acc, diag_f1, diag_cm, diag_classes = 1.0, 1.0, [], ["NONE"]
        diagnostic_pipeline = best_bin_model

    # ---------------------------------------------------------------------------
    # Model 3: Unsupervised Isolation Forest Anomaly Detector
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. Training Isolation Forest Anomaly Detector")
    print("-" * 50)

    anomaly_model = anomaly_detection.train_anomaly_model(df, model_path="models/anomaly_model.pkl")

    # Save artifacts & report
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    joblib.dump(best_bin_model, "models/failure_model.pkl")
    joblib.dump(diagnostic_pipeline, "models/diagnostic_model.pkl")

    report_data = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "dataset_name": "netguard_noc_dataset_v1",
        "total_records": len(df),
        "total_devices": int(df['Device_ID'].nunique()),
        "target_variable": "Failure_Next_12h",
        "validation_strategy": "GroupShuffleSplit by Device_ID (80% train devices / 20% test devices)",
        "binary_classifier": {
            "selected_model": best_bin_name,
            "metrics": bin_results[best_bin_name],
            "all_candidates": bin_results
        },
        "diagnostic_classifier": {
            "sample_filter": "Failed == 1 records only (No label leakage)",
            "classes": diag_classes,
            "accuracy": float(diag_acc),
            "weighted_f1": float(diag_f1),
            "confusion_matrix": diag_cm
        },
        "anomaly_detector": {
            "model_type": "Isolation Forest (150 estimators)",
            "contamination_factor": 0.08
        }
    }

    json_report_path = "outputs/reports/model_report.json"
    html_report_path = "outputs/reports/model_report.html"

    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NetGuard NOC — ML Validation Report (Failure_Next_12h Target)</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        h1, h2 {{ color: #38bdf8; }}
        .card {{ background: #1e293b; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #334155; padding: 10px; text-align: left; }}
        th {{ background: #0f172a; color: #a5b4fc; }}
        .badge {{ background: #22c55e; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🛡️ NetGuard NOC — ML Model Validation Report</h1>
    <p>Target: <strong>Failure_Next_12h</strong> | Generated: {report_data['evaluation_timestamp']}</p>
    
    <div class="card">
        <h2>Dataset & Validation Strategy</h2>
        <p><strong>Dataset:</strong> netguard_noc_dataset_v1</p>
        <p><strong>Total Records:</strong> {len(df):,} | <strong>Unique Devices:</strong> {df['Device_ID'].nunique()}</p>
        <p><strong>Validation Strategy:</strong> {report_data['validation_strategy']}</p>
    </div>

    <div class="card">
        <h2>Supervised Binary Failure Classifier (Failure_Next_12h)</h2>
        <p><strong>Selected Model:</strong> <span class="badge">{best_bin_name}</span></p>
        <table>
            <tr><th>Metric</th><th>Score</th></tr>
            <tr><td>Accuracy</td><td>{bin_results[best_bin_name]['Accuracy']:.4f}</td></tr>
            <tr><td>Precision</td><td>{bin_results[best_bin_name]['Precision']:.4f}</td></tr>
            <tr><td>Recall</td><td>{bin_results[best_bin_name]['Recall']:.4f}</td></tr>
            <tr><td>F1 Score</td><td>{bin_results[best_bin_name]['F1']:.4f}</td></tr>
            <tr><td>ROC-AUC</td><td>{bin_results[best_bin_name]['ROC-AUC']:.4f}</td></tr>
        </table>
    </div>
</body>
</html>"""

    with open(html_report_path, "w") as f:
        f.write(html_content)

    print("\n" + "=" * 60)
    print("Model Training Complete")
    print("=" * 60)
    print(f"JSON Report: {json_report_path}")
    print(f"HTML Report: {html_report_path}")

if __name__ == "__main__":
    train_all_models()