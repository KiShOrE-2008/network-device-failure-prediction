import os
import sys
import pandas as pd
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from target_engineering import compute_failure_next_12h_target


def test_future_target_does_not_use_current_failure():
    """
    Construct a deterministic device timeline to verify Failure_Next_12h target:
    1. Does NOT use current Failed(t) value.
    2. Correctly flags 1.0 if Failed == 1 occurs anywhere in t+1 ... t+12.
    3. Returns NaN for the final 12 boundary timesteps.
    """
    # 25 timesteps for DEV-001
    df = pd.DataFrame({
        "Device_ID": ["DEV-001"] * 25,
        "Timestamp": pd.date_range("2026-01-01", periods=25, freq="h"),
        "Failed": [0] * 25
    })

    # Set failure ONLY at index 10 (t=10)
    df.loc[10, "Failed"] = 1

    df_target = compute_failure_next_12h_target(df, horizon=12)

    # For t=0: future horizon t=1..12 contains t=10 (Failed=1) -> target must be 1.0
    assert df_target.loc[0, "Failure_Next_12h"] == 1.0

    # For t=9: future horizon t=10..21 contains t=10 (Failed=1) -> target must be 1.0
    assert df_target.loc[9, "Failure_Next_12h"] == 1.0

    # For t=10: current Failed=1, but future horizon t=11..22 contains no failures -> target must be 0.0
    # Crucial test: target at t=10 MUST NOT be 1.0 derived from current t=10 Failed state!
    assert df_target.loc[10, "Failure_Next_12h"] == 0.0

    # For t=11 and t=12: future window has no failures -> target must be 0.0
    assert df_target.loc[11, "Failure_Next_12h"] == 0.0
    assert df_target.loc[12, "Failure_Next_12h"] == 0.0

    # For the final 12 boundary rows (indices 13 to 24), future window is incomplete -> target must be NaN
    for idx in range(13, 25):
        assert np.isnan(df_target.loc[idx, "Failure_Next_12h"]), f"Index {idx} should be NaN"


def test_failure_next_12h_uses_same_device():
    """Verify target calculation does not leak failure events across device boundaries."""
    df_a = pd.DataFrame({
        "Device_ID": ["DEV-A"] * 20,
        "Timestamp": pd.date_range("2026-01-01", periods=20, freq="h"),
        "Failed": [0] * 20
    })

    df_b = pd.DataFrame({
        "Device_ID": ["DEV-B"] * 20,
        "Timestamp": pd.date_range("2026-01-01", periods=20, freq="h"),
        "Failed": [1] * 20  # DEV-B constantly failing
    })

    df = pd.concat([df_a, df_b], ignore_index=True)
    df_target = compute_failure_next_12h_target(df, horizon=12)

    # DEV-A rows 0 to 7 should be 0.0 because DEV-A has no failures
    dev_a_targets = df_target[df_target["Device_ID"] == "DEV-A"].reset_index(drop=True)
    for idx in range(0, 8):
        assert dev_a_targets.loc[idx, "Failure_Next_12h"] == 0.0, f"DEV-A index {idx} leaked failure from DEV-B"
