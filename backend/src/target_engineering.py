"""
src/target_engineering.py
--------------------------
Target Engineering & Validation utilities for NetGuard NOC.
Calculates Failure_Next_12h multi-step temporal horizon targets without target leakage.
"""

import pandas as pd
import numpy as np

# Standard forecasting target lookahead and purge horizon step count
PURGE_HORIZON = 12


def compute_failure_next_12h_target(df: pd.DataFrame, horizon: int = PURGE_HORIZON) -> pd.DataFrame:
    """
    Computes Failure_Next_12h target for time-series observations.

    Rules:
    1. Failure_Next_12h(t) = 1.0 if Failed == 1 occurs at any time t+1 ... t+12 for the SAME device.
    2. Must NOT include current Failed(t) or current Failure_Type(t).
    3. Must NOT peek beyond t+12.
    4. The final `horizon` (12) observations for each device have an incomplete future window
       and MUST be assigned NaN so they are excluded from supervised training/evaluation.
    """
    df = df.copy()
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.sort_values(['Device_ID', 'Timestamp']).reset_index(drop=True)

    grouped = df.groupby('Device_ID')

    # Construct matrix of future shifts: t+1, t+2, ..., t+12 for the SAME device
    future_shifts = []
    for shift_step in range(1, horizon + 1):
        s = grouped['Failed'].shift(-shift_step)
        future_shifts.append(s)

    future_matrix = pd.concat(future_shifts, axis=1)

    # Valid window requires all future 12 steps to belong to the same device (not NaN)
    has_valid_window = future_matrix.notnull().all(axis=1)
    any_future_failure = (future_matrix == 1).any(axis=1)

    # Assign 1.0 / 0.0 for valid windows, NaN for incomplete boundary rows
    target_values = np.where(has_valid_window, any_future_failure.astype(float), np.nan)
    df['Failure_Next_12h'] = target_values

    return df


def create_temporal_target_splits(
    df: pd.DataFrame, train_ratio: float = 0.8, horizon: int = PURGE_HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes leakage-free temporal target splitting.
    
    Flow:
    1. Sorts chronologically per device.
    2. Performs 80/20 temporal split on raw time-series per device FIRST.
    3. Computes target Failure_Next_12h separately on train and test partitions.
       This automatically purges the final `horizon` (12) boundary observations from the train set
       for each device, guaranteeing that NO training target uses failures from the test period.
    4. Filters out incomplete horizon NaN target rows.
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

    train_raw = df.iloc[train_indices].copy().reset_index(drop=True)
    test_raw = df.iloc[test_indices].copy().reset_index(drop=True)

    train_with_target = compute_failure_next_12h_target(train_raw, horizon=horizon)
    test_with_target = compute_failure_next_12h_target(test_raw, horizon=horizon)

    train_df = train_with_target[train_with_target['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    train_df['Failure_Next_12h'] = train_df['Failure_Next_12h'].astype(int)

    test_df = test_with_target[test_with_target['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    test_df['Failure_Next_12h'] = test_df['Failure_Next_12h'].astype(int)

    return train_df, test_df


def create_temporal_target_splits_3way(
    df: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2, horizon: int = PURGE_HORIZON
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes 60/20/20 chronological Train/Validation/Test target splitting per device.
    
    Guarantees:
    - Train (60%) -> Validation (20%) -> Test (20%) chronologically per device.
    - Purges 12 boundary steps at Train/Validation boundary and Validation/Test boundary.
    - Zero future leakage across partitions.
    """
    if 'Timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df = df.sort_values(['Device_ID', 'Timestamp']).reset_index(drop=True)

    train_indices = []
    val_indices = []
    test_indices = []

    for _, group in df.groupby('Device_ID', sort=False):
        n = len(group)
        n_train = max(1, int(np.floor(n * train_ratio)))
        n_val = max(1, int(np.floor(n * val_ratio)))
        indices = group.index.values
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    train_raw = df.iloc[train_indices].copy().reset_index(drop=True)
    val_raw = df.iloc[val_indices].copy().reset_index(drop=True)
    test_raw = df.iloc[test_indices].copy().reset_index(drop=True)

    train_with_target = compute_failure_next_12h_target(train_raw, horizon=horizon)
    val_with_target = compute_failure_next_12h_target(val_raw, horizon=horizon)
    test_with_target = compute_failure_next_12h_target(test_raw, horizon=horizon)

    train_df = train_with_target[train_with_target['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    train_df['Failure_Next_12h'] = train_df['Failure_Next_12h'].astype(int)

    val_df = val_with_target[val_with_target['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    val_df['Failure_Next_12h'] = val_df['Failure_Next_12h'].astype(int)

    test_df = test_with_target[test_with_target['Failure_Next_12h'].notnull()].copy().reset_index(drop=True)
    test_df['Failure_Next_12h'] = test_df['Failure_Next_12h'].astype(int)

    return train_df, val_df, test_df


