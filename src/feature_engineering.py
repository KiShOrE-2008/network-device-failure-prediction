import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    'CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors',
    'Packet_Loss', 'Bandwidth_Usage', 'Uptime', 'Log_Errors',
    'Syslog_Critical_Count', 'CPU_5step_avg', 'CPU_Trend', 'CPU_Spike',
    'Memory_5step_avg', 'Memory_Trend', 'Temperature_5step_avg',
    'Temperature_Trend', 'Temperature_Spike', 'Error_5step_avg',
    'Error_Trend', 'Error_Spike', 'PacketLoss_5step_avg', 'PacketLoss_Trend'
]

METADATA_COLUMNS = [
    'Device_ID', 'Hostname', 'IP_Address', 'Device_Type', 'Vendor',
    'Model', 'Location', 'Rack', 'Firmware', 'Timestamp', 'Latest_Syslog'
]

EXCLUDE_COLUMNS = ['Hidden_Degradation_State', 'Failed', 'Failure_Type', 'Failure_Next_12h']


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 5-step rolling averages, trends, and spikes on telemetry data sorted by Device_ID and Timestamp.
    Uses past observations only to prevent temporal leakage.
    """
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(['Device_ID', 'Timestamp']).reset_index(drop=True)

    # Calculate rolling features per device
    grouped = df.groupby('Device_ID')

    for col in ['CPU_Usage', 'Memory_Usage', 'Temperature', 'Interface_Errors', 'Packet_Loss']:
        short_name = col.replace('_Usage', '')
        if f'{short_name}_5step_avg' not in df.columns or df[f'{short_name}_5step_avg'].isnull().all():
            df[f'{short_name}_5step_avg'] = grouped[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        if f'{short_name}_Trend' not in df.columns or df[f'{short_name}_Trend'].isnull().all():
            df[f'{short_name}_Trend'] = grouped[col].transform(lambda x: x.diff(1))

    # Calculate Spike flags if not present
    if 'CPU_Spike' not in df.columns or df['CPU_Spike'].isnull().all():
        df['CPU_Spike'] = ((df['CPU_Usage'] > 85) & (df['CPU_Trend'] > 15)).astype(int)

    if 'Temperature_Spike' not in df.columns or df['Temperature_Spike'].isnull().all():
        df['Temperature_Spike'] = ((df['Temperature'] > 75) & (df['Temperature_Trend'] > 5)).astype(int)

    if 'Error_Spike' not in df.columns or df['Error_Spike'].isnull().all():
        df['Error_Spike'] = ((df['Interface_Errors'] > 20) & (df['Error_Trend'] > 10)).astype(int)

    # Fill remaining NaNs with defaults
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list = None):
    """
    Extracts numerical feature matrix X for ML models.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)
    return X
