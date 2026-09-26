import os
import pandas as pd
import pytest


def test_dataset_v1_files_exist():
    base_dir = "data/netguard_noc_dataset_v1"
    assert os.path.exists(os.path.join(base_dir, "network_devices_timeseries.csv"))
    assert os.path.exists(os.path.join(base_dir, "network_devices.csv"))
    assert os.path.exists(os.path.join(base_dir, "failure_events.csv"))
    assert os.path.exists(os.path.join(base_dir, "topology.csv"))


def test_device_count_and_schema():
    csv_path = "data/netguard_noc_dataset_v1/network_devices.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 500
    assert "Device_ID" in df.columns
    assert "Vendor" in df.columns
    assert "Model" in df.columns


def test_timeseries_target_columns():
    csv_path = "data/netguard_noc_dataset_v1/network_devices_timeseries.csv"
    df = pd.read_csv(csv_path, nrows=100)
    assert "Failed" in df.columns
    assert "Failure_Next_12h" in df.columns
    assert "Failure_Type" in df.columns
    assert "Hidden_Degradation_State" in df.columns


def test_hidden_degradation_not_in_features():
    from feature_engineering import FEATURE_COLUMNS, EXCLUDE_COLUMNS
    assert "Hidden_Degradation_State" not in FEATURE_COLUMNS
    assert "Hidden_Degradation_State" in EXCLUDE_COLUMNS
