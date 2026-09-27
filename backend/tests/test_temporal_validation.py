import os
import sys
import pandas as pd
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train_model import temporal_split_per_device
from target_engineering import create_temporal_target_splits_3way


def test_train_test_are_temporally_separated():
    """
    Verify that temporal_split_per_device splits every device strictly chronologically
    such that max(train timestamp) < min(test timestamp) for every device.
    """
    devices = ["DEV-001", "DEV-002", "DEV-003"]
    records = []

    for dev in devices:
        ts_range = pd.date_range("2026-01-01", periods=50, freq="h")
        for ts in ts_range:
            records.append({
                "Device_ID": dev,
                "Timestamp": ts,
                "CPU_Usage": np.random.uniform(10, 50)
            })

    df = pd.DataFrame(records)
    train_df, test_df = temporal_split_per_device(df, train_ratio=0.8)

    for dev in devices:
        dev_tr = train_df[train_df["Device_ID"] == dev]
        dev_te = test_df[test_df["Device_ID"] == dev]

        assert len(dev_tr) > 0, f"No train records for {dev}"
        assert len(dev_te) > 0, f"No test records for {dev}"

        max_train_ts = dev_tr["Timestamp"].max()
        min_test_ts = dev_te["Timestamp"].min()

        assert max_train_ts < min_test_ts, f"Device {dev} temporal violation: max train {max_train_ts} >= min test {min_test_ts}"


def test_no_future_timestamp_in_training():
    """Verify that no training timestamp is greater than or equal to any test timestamp for a device."""
    df = pd.DataFrame({
        "Device_ID": ["DEV-001"] * 10,
        "Timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "Value": list(range(10))
    })

    train_df, test_df = temporal_split_per_device(df, train_ratio=0.8)

    # 80% of 10 = 8 rows (indices 0 to 7)
    assert len(train_df) == 8
    assert len(test_df) == 2

    assert list(train_df["Value"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(test_df["Value"]) == [8, 9]
    assert train_df["Timestamp"].max() < test_df["Timestamp"].min()


def test_create_temporal_target_splits_3way_chronological_isolation():
    """Verify 60/20/20 chronological train < val < test isolation."""
    df = pd.DataFrame({
        "Device_ID": ["DEV-001"] * 100,
        "Timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
        "Failed": [0] * 100
    })

    train_df, val_df, test_df = create_temporal_target_splits_3way(df, train_ratio=0.6, val_ratio=0.2, horizon=12)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    assert train_df["Timestamp"].max() < val_df["Timestamp"].min()
    assert val_df["Timestamp"].max() < test_df["Timestamp"].min()

