"""
src/feature_engineering.py
---------------------------
Authoritative Feature Schema & Leakage-Safe Feature Engineering Engine for NetGuard NOC.

Defines standard feature matrices, past-only temporal rolling window transformations,
and schema constants used across training, batch prediction, and model metadata.
"""

import pandas as pd
import numpy as np

# Single Source of Truth for Model Features
CATEGORICAL_FEATURES = ["Device_Type"]

NUMERICAL_FEATURES = [
    'CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors',
    'Packet_Loss', 'Bandwidth_Usage', 'Uptime', 'Log_Errors',
    'Syslog_Critical_Count', 'CPU_5step_avg', 'CPU_Trend', 'CPU_Spike',
    'Memory_5step_avg', 'Memory_Trend', 'Temperature_5step_avg',
    'Temperature_Trend', 'Temperature_Spike', 'Error_5step_avg',
    'Error_Trend', 'Error_Spike', 'PacketLoss_5step_avg', 'PacketLoss_Trend'
]

# Combined Feature Matrix Schema
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Diagnostic Feature Schema (used by root cause multi-class classifier)
DIAGNOSTIC_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()

# Metadata & Identifiers
METADATA_COLUMNS = [
    'Device_ID', 'Hostname', 'IP_Address', 'Device_Type', 'Vendor',
    'Model', 'Location', 'Rack', 'Firmware', 'Timestamp', 'Latest_Syslog'
]

# Explicitly Excluded Fields (latent state, ground truth targets & future indicators)
EXCLUDE_COLUMNS = [
    'Hidden_Degradation_State',
    'Latent_Health_State',
    'Failed',
    'Failure_Type',
    'Failure_Next_12h'
]


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 5-step rolling averages, trends, and spikes on telemetry data sorted by Device_ID and Timestamp.
    Uses strictly past observations (shift(1)) to guarantee zero future data leakage.
    """
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['Device_ID', 'Timestamp']).reset_index(drop=True)

    # Group by Device_ID for device-isolated time-series calculations
    grouped = df.groupby('Device_ID')

    name_map = {
        'CPU_Usage': 'CPU',
        'Memory_Usage': 'Memory',
        'Temperature': 'Temperature',
        'Interface_Errors': 'Error',
        'Packet_Loss': 'PacketLoss'
    }

    for col, short_name in name_map.items():
        if col in df.columns:
            # 5-step rolling average using past observations only (shift(1))
            avg_col = f'{short_name}_5step_avg'
            df[avg_col] = grouped[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0.0)

            # 1-step trend (diff from previous step)
            trend_col = f'{short_name}_Trend'
            df[trend_col] = grouped[col].transform(lambda x: x.diff(1)).fillna(0.0)

    # Spike indicators calculated against past rolling average and trend
    cpu_usage = df['CPU_Usage'] if 'CPU_Usage' in df.columns else 0
    cpu_trend = df['CPU_Trend'] if 'CPU_Trend' in df.columns else 0
    temp_usage = df['Temperature'] if 'Temperature' in df.columns else 0
    temp_trend = df['Temperature_Trend'] if 'Temperature_Trend' in df.columns else 0
    err_usage = df['Interface_Errors'] if 'Interface_Errors' in df.columns else 0
    err_trend = df['Error_Trend'] if 'Error_Trend' in df.columns else 0

    df['CPU_Spike'] = ((cpu_usage > 85) & (cpu_trend > 15)).astype(int)
    df['Temperature_Spike'] = ((temp_usage > 75) & (temp_trend > 5)).astype(int)
    df['Error_Spike'] = ((err_usage > 20) & (err_trend > 10)).astype(int)

    # Fill remaining NaNs with defaults
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """
    Extracts numerical and categorical feature matrix X for ML model training and inference.
    Ensures no excluded latent variables or future targets are present.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS.copy()

    # Strict audit: verify no EXCLUDE_COLUMNS are present in feature_cols
    forbidden = [c for c in feature_cols if c in EXCLUDE_COLUMNS]
    if forbidden:
        raise ValueError(f"CRITICAL ERROR: Leakage detected! Feature schema contains excluded columns: {forbidden}")

    X = df.copy()
    for col in feature_cols:
        if col not in X.columns:
            if col in CATEGORICAL_FEATURES:
                X[col] = "Router"
            else:
                X[col] = 0.0

    X = X[feature_cols].copy()
    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

    return X
