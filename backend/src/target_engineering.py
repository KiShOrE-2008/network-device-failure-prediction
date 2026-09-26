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
