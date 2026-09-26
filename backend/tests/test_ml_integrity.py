"""
tests/test_ml_integrity.py
---------------------------
Automated Data & Feature Leakage Integrity Test Suite for NetGuard NOC.
Enforces the 5 Hard Guarantees against temporal and target leakage in the ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import (
    FEATURE_COLUMNS,
    EXCLUDE_COLUMNS,
    compute_rolling_features,
    prepare_feature_matrix
)
from target_engineering import compute_failure_next_12h_target, PURGE_HORIZON


def test_leakage_check_1_hidden_degradation_state_excluded():
    """Test 1: Hidden_Degradation_State cannot enter feature matrix."""
    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Hidden_Degradation_State': [0, 1, 1, 1, 1]
    })
    df = compute_rolling_features(df)
    
    # Attempting to pass schema with Hidden_Degradation_State must raise ValueError
    with pytest.raises(ValueError, match="Leakage detected"):
        prepare_feature_matrix(df, feature_cols=FEATURE_COLUMNS + ['Hidden_Degradation_State'])
    
    # Default feature matrix must not contain it
    X = prepare_feature_matrix(df)
    assert 'Hidden_Degradation_State' not in X.columns


def test_leakage_check_2_failed_excluded():
    """Test 2: Failed cannot enter feature matrix."""
    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Failed': [0, 0, 1, 0, 0]
    })
    df = compute_rolling_features(df)

    with pytest.raises(ValueError, match="Leakage detected"):
        prepare_feature_matrix(df, feature_cols=FEATURE_COLUMNS + ['Failed'])

    X = prepare_feature_matrix(df)
    assert 'Failed' not in X.columns


def test_leakage_check_3_failure_type_excluded():
    """Test 3: Failure_Type cannot enter feature matrix."""
    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Failure_Type': ['NONE', 'NONE', 'THERMAL', 'NONE', 'NONE']
    })
    df = compute_rolling_features(df)

    with pytest.raises(ValueError, match="Leakage detected"):
        prepare_feature_matrix(df, feature_cols=FEATURE_COLUMNS + ['Failure_Type'])

    X = prepare_feature_matrix(df)
    assert 'Failure_Type' not in X.columns


def test_leakage_check_4_failure_next_12h_excluded():
    """Test 4: Failure_Next_12h cannot enter feature matrix."""
    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Failure_Next_12h': [1.0, 1.0, 0.0, 0.0, 0.0]
    })
    df = compute_rolling_features(df)

    with pytest.raises(ValueError, match="Leakage detected"):
        prepare_feature_matrix(df, feature_cols=FEATURE_COLUMNS + ['Failure_Next_12h'])

    X = prepare_feature_matrix(df)
    assert 'Failure_Next_12h' not in X.columns


def test_leakage_check_5_future_observations_do_not_affect_rolling_features():
    """Test 5: Future observations cannot affect past rolling features (shift(1) insulation)."""
    # Create baseline telemetry
    df1 = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    feat1 = compute_rolling_features(df1)
    avg_t3_before = feat1.loc[2, 'CPU_5step_avg']

    # Modify future observation at t=4 and t=5
    df2 = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 5,
        'Timestamp': pd.date_range('2026-01-01', periods=5, freq='1h'),
        'CPU_Usage': [10.0, 20.0, 30.0, 999.0, 999.0]
    })
    feat2 = compute_rolling_features(df2)
    avg_t3_after = feat2.loc[2, 'CPU_5step_avg']

    # Rolling average at t=3 must be 100% identical regardless of future values at t=4/5
    assert avg_t3_before == avg_t3_after


def test_leakage_check_6_target_does_not_use_current_failed():
    """Test 6: Failure_Next_12h(t) must NOT equal 1 if Failed(t) = 1 but Failed(t+1..t+12) = 0."""
    timestamps = pd.date_range('2026-01-01', periods=20, freq='1h')
    failed_series = [0] * 20
    failed_series[2] = 1  # Failed at index 2 (t=2)

    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 20,
        'Timestamp': timestamps,
        'Failed': failed_series
    })

    df_target = compute_failure_next_12h_target(df, horizon=12)

    # At index 2, Failed(t=2) == 1, but t+1..t+12 are all 0.
    # Therefore, Failure_Next_12h(t=2) MUST equal 0.0, NOT 1.0!
    assert df_target.loc[2, 'Failure_Next_12h'] == 0.0

    # At index 0 and 1, t+1..t+12 contains index 2 where Failed == 1.
    # Therefore, Failure_Next_12h(t=0) and Failure_Next_12h(t=1) MUST equal 1.0.
    assert df_target.loc[0, 'Failure_Next_12h'] == 1.0
    assert df_target.loc[1, 'Failure_Next_12h'] == 1.0


def test_leakage_check_7_final_12_rows_have_nan_target():
    """Test 7: Final 12 rows per device have incomplete lookahead window and MUST be assigned NaN target."""
    timestamps = pd.date_range('2026-01-01', periods=25, freq='1h')
    df = pd.DataFrame({
        'Device_ID': ['DEV-001'] * 25,
        'Timestamp': timestamps,
        'Failed': [0] * 25
    })

    df_target = compute_failure_next_12h_target(df, horizon=PURGE_HORIZON)

    # The last 12 rows (index 13 to 24) must have NaN target
    assert df_target.iloc[-12:]['Failure_Next_12h'].isnull().all()
    # First 13 rows (index 0 to 12) have full lookahead windows and valid non-null targets
    assert df_target.iloc[:13]['Failure_Next_12h'].notnull().all()
