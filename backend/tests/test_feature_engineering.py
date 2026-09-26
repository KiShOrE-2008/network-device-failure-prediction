import os
import sys
import pandas as pd
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    EXCLUDE_COLUMNS,
    compute_rolling_features,
    prepare_feature_matrix
)


def test_hidden_degradation_not_in_features():
    """Verify simulation latent states and target labels are strictly excluded from features."""
    for col in EXCLUDE_COLUMNS:
        assert col not in FEATURE_COLUMNS, f"Forbidden column {col} found in FEATURE_COLUMNS"
        assert col not in NUMERICAL_FEATURES, f"Forbidden column {col} found in NUMERICAL_FEATURES"
        assert col not in CATEGORICAL_FEATURES, f"Forbidden column {col} found in CATEGORICAL_FEATURES"

    assert "Hidden_Degradation_State" not in FEATURE_COLUMNS
    assert "Latent_Health_State" not in FEATURE_COLUMNS
    assert "Failed" not in FEATURE_COLUMNS
    assert "Failure_Next_12h" not in FEATURE_COLUMNS


def test_feature_schema_matches_model():
    """Verify feature matrix contains expected columns and prevents feature leakage."""
    sample_df = pd.DataFrame({
        "Device_ID": ["DEV-001"],
        "Timestamp": ["2026-01-01 00:00:00"],
        "Device_Type": ["Router"],
        "CPU_Usage": [50.0],
        "Hidden_Degradation_State": [10.0],
        "Failed": [1]
    })
    
    X = prepare_feature_matrix(sample_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "Hidden_Degradation_State" not in X.columns
    assert "Failed" not in X.columns

    with pytest.raises(ValueError, match="Leakage detected"):
        prepare_feature_matrix(sample_df, feature_cols=["CPU_Usage", "Hidden_Degradation_State"])


def test_rolling_features_are_past_only():
    """Verify rolling averages use shift(1) so current/future value changes do not affect past rolling avg."""
    df = pd.DataFrame({
        "Device_ID": ["DEV-001"] * 5,
        "Timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        "CPU_Usage": [10.0, 20.0, 30.0, 100.0, 100.0],
        "Memory_Usage": [10.0, 10.0, 10.0, 10.0, 10.0],
        "Temperature": [30.0, 30.0, 30.0, 30.0, 30.0],
        "Interface_Errors": [0, 0, 0, 0, 0],
        "Packet_Loss": [0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    df_rolled = compute_rolling_features(df)
    
    # Row 0: shift(1) is NaN -> 0.0
    assert df_rolled.loc[0, "CPU_5step_avg"] == 0.0
    # Row 1: shift(1) is [10.0] -> avg = 10.0
    assert df_rolled.loc[1, "CPU_5step_avg"] == 10.0
    # Row 2: shift(1) is [10.0, 20.0] -> avg = 15.0
    assert df_rolled.loc[2, "CPU_5step_avg"] == 15.0
    # Row 3: shift(1) is [10.0, 20.0, 30.0] -> avg = 20.0 (independent of row 3 CPU_Usage=100.0)
    assert df_rolled.loc[3, "CPU_5step_avg"] == 20.0
