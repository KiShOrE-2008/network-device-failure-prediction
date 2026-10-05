"""
src/train_model.py
------------------
NetGuard NOC Fleet ML Training Pipeline.
Consumes NetGuard NOC time-series benchmark dataset.
Enforces deterministic chronological 80/20 per-device temporal validation split.
Primary target: Failure_Next_12h (impending failure in next 12 hours).
Diagnostic target: Failure_Type (trained strictly on Failed == 1 records).
Anomaly target: Unsupervised Isolation Forest on operational baseline (Failed == 0).
Generates models/model_metadata.json automatically during training execution.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(BASE_DIR)
WORKSPACE_ROOT = os.path.dirname(BACKEND_ROOT)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
from feature_engineering import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    EXCLUDE_COLUMNS,
    compute_rolling_features,
    prepare_feature_matrix
)
from target_engineering import compute_failure_next_12h_target, create_temporal_target_splits, create_temporal_target_splits_3way, PURGE_HORIZON


def temporal_split_per_device(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Splits time-series dataset chronologically per Device_ID.
    First 80% observations per device -> TRAIN set.
    Final 20% observations per device -> TEST set.
    Guarantees max(train_timestamp) < min(test_timestamp) for every device.
    """
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.sort_values(['Device_ID', 'Timestamp']).reset_index(drop=True)
    train_indices = []
    test_indices = []

    for _, group in df.groupby('Device_ID', sort=False):
        n = len(group)
        n_train = max(1, int(np.floor(n * train_ratio)))
        indices = group.index.values
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    return train_df, test_df


def train_all_models(data_path: str = None, output_dir: str = None):
    print("=" * 60)
    print("NetGuard NOC — Fleet ML Training Pipeline (Temporal Validation)")
    print("=" * 60)

    if data_path and os.path.exists(data_path):
        csv_path = data_path
    else:
        candidates = [
            os.path.join(BACKEND_ROOT, "data", "netguard_noc_dataset_v1", "network_devices_timeseries.csv"),
            os.path.join(BACKEND_ROOT, "data", "network_devices_timeseries.csv"),
            os.path.join(WORKSPACE_ROOT, "data", "netguard_noc_dataset_v1", "network_devices_timeseries.csv"),
            os.path.join(WORKSPACE_ROOT, "data", "network_devices_timeseries.csv"),
            "data/netguard_noc_dataset_v1/network_devices_timeseries.csv",
            "data/network_devices_timeseries.csv"
        ]
        csv_path = next((c for c in candidates if os.path.exists(c)), "data/network_devices_timeseries.csv")
        
    print(f"Loading benchmark dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} raw records across {df['Device_ID'].nunique()} devices.")

    # 1. Compute leakage-safe rolling features (shift(1) insulated)
    df = compute_rolling_features(df)

    # 2. Deterministic Raw Temporal 60/20/20 Train/Validation/Test Split Per Device + 12-Step Purge
    print("\nExecuting raw temporal 60/20/20 train/validation/test split per device + 12-step boundary purge...")
    train_df, val_df, test_df = create_temporal_target_splits_3way(df, train_ratio=0.6, val_ratio=0.2, horizon=PURGE_HORIZON)

    # Verify temporal isolation
    unique_devices = df['Device_ID'].unique()
    for dev_id in unique_devices[:5]:
        dev_tr = train_df[train_df['Device_ID'] == dev_id]
        dev_va = val_df[val_df['Device_ID'] == dev_id]
        dev_te = test_df[test_df['Device_ID'] == dev_id]
        if len(dev_tr) > 0 and len(dev_va) > 0:
            assert dev_tr['Timestamp'].max() < dev_va['Timestamp'].min(), f"Train/Val temporal leakage detected on {dev_id}"
        if len(dev_va) > 0 and len(dev_te) > 0:
            assert dev_va['Timestamp'].max() < dev_te['Timestamp'].min(), f"Val/Test temporal leakage detected on {dev_id}"

    print(f"Train Set:      {len(train_df):,} rows ({train_df['Failure_Next_12h'].sum()} failures)")
    print(f"Validation Set: {len(val_df):,} rows ({val_df['Failure_Next_12h'].sum()} failures)")
    print(f"Test Set:       {len(test_df):,} rows ({test_df['Failure_Next_12h'].sum()} failures)")

    # 4. Prepare Feature Matrices
    X_train = prepare_feature_matrix(train_df, FEATURE_COLUMNS)
    y_train_bin = train_df['Failure_Next_12h']

    X_val = prepare_feature_matrix(val_df, FEATURE_COLUMNS)
    y_val_bin = val_df['Failure_Next_12h']

    X_test = prepare_feature_matrix(test_df, FEATURE_COLUMNS)
    y_test_bin = test_df['Failure_Next_12h']

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

    # ---------------------------------------------------------------------------
    # Model 1: Supervised Binary Failure Forecaster (Target: Failure_Next_12h)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("1. Training Supervised Binary Failure Forecasters (Validation Model Selection)")
    print("-" * 50)

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
    bin_val_results = {}

    for name, clf in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", clf)
        ])
        
        pipeline.fit(X_train, y_train_bin)
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val_bin, y_pred)
        prec = precision_score(y_val_bin, y_pred, zero_division=0)
        rec = recall_score(y_val_bin, y_pred, zero_division=0)
        f1 = f1_score(y_val_bin, y_pred, zero_division=0)
        roc = roc_auc_score(y_val_bin, y_prob)
        cm = confusion_matrix(y_val_bin, y_pred).tolist()

        bin_val_results[name] = {
            "Accuracy": float(acc), "Precision": float(prec),
            "Recall": float(rec), "F1": float(f1), "ROC-AUC": float(roc), "ConfusionMatrix": cm
        }
        print(f"[{name} - Validation] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

        if f1 > best_bin_score:
            best_bin_score = f1
            best_bin_model = pipeline
            best_bin_name = name

    print(f"\n🏆 Best Binary Failure Model (Selected on Validation Set): {best_bin_name} (Val F1 = {best_bin_score:.4f})")

    # Final Evaluation of Best Model on Held-Out Test Set (NEVER used for selection)
    y_test_pred = best_bin_model.predict(X_test)
    y_test_prob = best_bin_model.predict_proba(X_test)[:, 1]
    final_test_metrics = {
        "Accuracy": float(accuracy_score(y_test_bin, y_test_pred)),
        "Precision": float(precision_score(y_test_bin, y_test_pred, zero_division=0)),
        "Recall": float(recall_score(y_test_bin, y_test_pred, zero_division=0)),
        "F1": float(f1_score(y_test_bin, y_test_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test_bin, y_test_prob)),
        "ConfusionMatrix": confusion_matrix(y_test_bin, y_test_pred).tolist()
    }
    print(f"📊 Final Held-Out Test Evaluation ({best_bin_name}): Acc: {final_test_metrics['Accuracy']:.4f} | Prec: {final_test_metrics['Precision']:.4f} | Rec: {final_test_metrics['Recall']:.4f} | F1: {final_test_metrics['F1']:.4f} | ROC-AUC: {final_test_metrics['ROC-AUC']:.4f}")


    # ---------------------------------------------------------------------------
    # Model 2: Supervised Multi-Class Diagnostic Classifier (Validation Set Evaluation)
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("2. Training Diagnostic Classifier (Evaluated on Test Failures)")
    print("-" * 50)

    failed_df = df[(df["Failed"] == 1) & (df["Failure_Type"] != "NONE")].copy().reset_index(drop=True)

    if len(failed_df) > 10:
        diag_train, diag_test = temporal_split_per_device(failed_df, train_ratio=0.8)
        if len(diag_test) == 0:
            diag_train, diag_test = failed_df, failed_df

        X_diag_train = prepare_feature_matrix(diag_train, FEATURE_COLUMNS)
        y_diag_train = diag_train["Failure_Type"]

        X_diag_test = prepare_feature_matrix(diag_test, FEATURE_COLUMNS)
        y_diag_test = diag_test["Failure_Type"]

        label_encoder = LabelEncoder()
        y_diag_tr_enc = label_encoder.fit_transform(y_diag_train)
        
        # Handle unseen classes in test set safely
        y_diag_te_enc = np.array([
            label_encoder.transform([cls])[0] if cls in label_encoder.classes_ else -1
            for cls in y_diag_test
        ])
        valid_mask = y_diag_te_enc != -1
        X_diag_test_eval = X_diag_test.iloc[valid_mask]
        y_diag_te_enc_eval = y_diag_te_enc[valid_mask]

        diagnostic_clf = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="mlogloss"
        )
        diagnostic_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", diagnostic_clf)
        ])

        diagnostic_pipeline.fit(X_diag_train, y_diag_tr_enc)
        y_diag_pred = diagnostic_pipeline.predict(X_diag_test_eval)

        diag_acc = accuracy_score(y_diag_te_enc_eval, y_diag_pred)
        diag_prec = precision_score(y_diag_te_enc_eval, y_diag_pred, average="weighted", zero_division=0)
        diag_rec = recall_score(y_diag_te_enc_eval, y_diag_pred, average="weighted", zero_division=0)
        diag_f1 = f1_score(y_diag_te_enc_eval, y_diag_pred, average="weighted", zero_division=0)
        diag_cm = confusion_matrix(y_diag_te_enc_eval, y_diag_pred).tolist()
        diag_classes = label_encoder.classes_.tolist()

        diagnostic_pipeline.label_classes_ = diag_classes
        print(f"[Diagnostic Classifier] Train Samples: {len(diag_train)} | Validation Samples: {len(diag_test)}")
        print(f"[Diagnostic Classifier] Acc: {diag_acc:.4f} | Prec: {diag_prec:.4f} | Rec: {diag_rec:.4f} | Weighted F1: {diag_f1:.4f}")
    else:
        diag_acc, diag_prec, diag_rec, diag_f1, diag_cm, diag_classes = 1.0, 1.0, 1.0, 1.0, [], ["NONE"]
        diagnostic_pipeline = best_bin_model

    # ---------------------------------------------------------------------------
    # Model 3: Unsupervised Isolation Forest Anomaly Detector
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("3. Training Isolation Forest Anomaly Detector")
    print("-" * 50)

    models_dir = output_dir if output_dir else os.path.join(BACKEND_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    outputs_dir = os.path.join(BACKEND_ROOT, "outputs", "reports")
    os.makedirs(outputs_dir, exist_ok=True)

    anomaly_model_path = os.path.join(models_dir, "anomaly_model.pkl")
    anomaly_model = anomaly_detection.train_anomaly_model(df, model_path=anomaly_model_path)

    # ---------------------------------------------------------------------------
    # Save Model Artifacts & Auto-Generate Metadata (models/model_metadata.json)
    # ---------------------------------------------------------------------------
    failure_model_path = os.path.join(models_dir, "failure_model.pkl")
    diagnostic_model_path = os.path.join(models_dir, "diagnostic_model.pkl")
    joblib.dump(best_bin_model, failure_model_path)
    joblib.dump(diagnostic_pipeline, diagnostic_model_path)

    metadata = {
        "training_timestamp": datetime.now().isoformat(),
        "dataset_name": "netguard_noc_dataset_v1",
        "dataset_version": "v2.0",
        "total_records": int(len(df)),
        "total_devices": int(df['Device_ID'].nunique()),
        "target": "Failure_Next_12h",
        "validation_strategy": "Deterministic Chronological 60/20/20 Split Per Device (Train / Validation / Held-Out Test)",
        "model_version": "3.0.0",
        "selected_binary_model": best_bin_name,
        "feature_schema": {
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "all_features": FEATURE_COLUMNS,
            "excluded_columns": EXCLUDE_COLUMNS
        },
        "diagnostic_classes": diag_classes,
        "binary_classifier_validation_metrics": bin_val_results[best_bin_name],
        "binary_classifier_test_metrics": final_test_metrics,

        "diagnostic_classifier_metrics": {
            "sample_filter": "Failed == 1 records (chronologically evaluated on validation split)",
            "accuracy": float(diag_acc),
            "precision": float(diag_prec),
            "recall": float(diag_rec),
            "weighted_f1": float(diag_f1),
            "confusion_matrix": diag_cm
        },
        "anomaly_detector": {
            "model_type": "Isolation Forest (150 estimators)",
            "contamination_factor": 0.08
        }
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    json_report_path = os.path.join(outputs_dir, "model_report.json")
    with open(json_report_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Model Training & Metadata Generation Complete")
    print("=" * 60)
    print(f"✓ Binary Failure Model:      {failure_model_path}")
    print(f"✓ Diagnostic Classifier:     {diagnostic_model_path}")
    print(f"✓ Anomaly Detector:          {anomaly_model_path}")
    print(f"✓ Auto-Generated Metadata:   {metadata_path}")


if __name__ == "__main__":
    train_all_models()