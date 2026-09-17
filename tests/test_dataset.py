import os
import sys
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from generate_dataset import generate_network_telemetry

def test_dataset_generation():
    generate_network_telemetry(num_devices=10, steps_per_device=20)
    timeseries_path = "data/network_devices_timeseries.csv"
    assert os.path.exists(timeseries_path)
    
    df = pd.read_csv(timeseries_path)
    assert len(df) == 200
    assert df["Device_ID"].nunique() == 10
    
    expected_cols = [
        "Timestamp", "Device_ID", "Hostname", "IP_Address", "Device_Type",
        "Vendor", "Model", "Location", "CPU_Usage", "Memory_Usage",
        "Temperature", "Failure_Type", "Failed", "CPU_Trend", "CPU_Spike"
    ]
    for col in expected_cols:
        assert col in df.columns
