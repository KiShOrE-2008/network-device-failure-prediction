import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv(
    "data/network_devices.csv"
)

print("Dataset loaded successfully.")

# --------------------------------
# Features
# --------------------------------

features = [
    "Device_Type",
    "CPU_Usage",
    "Memory_Usage",
    "Temperature",
    "Uptime",
    "Interface_Errors",
    "Packet_Loss",
    "Bandwidth_Usage",
    "Log_Errors"
]

X = df[features]

y = df["Failed"]


# --------------------------------
# Feature Types
# --------------------------------

categorical_features = [
    "Device_Type"
]

numerical_features = [
    "CPU_Usage",
    "Memory_Usage",
    "Temperature",
    "Uptime",
    "Interface_Errors",
    "Packet_Loss",
    "Bandwidth_Usage",
    "Log_Errors"
]


# --------------------------------
# Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_features
        )
    ]
)


# --------------------------------
# Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Models
# --------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
}


results = {}

best_model = None
best_score = 0
best_model_name = ""


# --------------------------------
# Train Models
# --------------------------------

for name, model in models.items():

    print("\n" + "=" * 50)

    print(
        f"Training {name}"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Select best model based on F1
    if f1 > best_score:

        best_score = f1

        best_model = pipeline

        best_model_name = name


# --------------------------------
# Results
# --------------------------------

print("\n" + "=" * 50)

print("MODEL COMPARISON")

print("=" * 50)

for model_name, metrics in results.items():

    print(
        f"\n{model_name}"
    )

    for metric, value in metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )


# --------------------------------
# Save Best Model
# --------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/failure_model.pkl"
)

print("\n" + "=" * 50)

print(
    f"Best Model: {best_model_name}"
)

print(
    f"Best F1 Score: {best_score:.4f}"
)

print(
    "Model saved to:"
)

print(
    "models/failure_model.pkl"
)