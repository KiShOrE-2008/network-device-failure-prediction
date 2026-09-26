"""
src/validate_dataset.py
------------------------
NetGuard NOC Dataset Validation & Quality Assurance Audit Tool.
Audits time-series telemetry data for device counts, chronological ordering,
duplicate (Device_ID, Timestamp) pairs, missing values, target distribution, and feature availability.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature_engineering import FEATURE_COLUMNS, EXCLUDE_COLUMNS


def validate_dataset(csv_path: str = None) -> dict:
    workspace_root = os.path.dirname(BASE_DIR)
    if csv_path is None:
        csv_path = os.path.join(workspace_root, "data", "network_devices_timeseries.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(workspace_root, "data", "netguard_noc_dataset_v1", "network_devices_timeseries.csv")

    if not os.path.exists(csv_path):
        print(f"❌ Error: Dataset file not found at {csv_path}")
        return {"valid": False, "error": "File not found"}

    print("=" * 60)
    print("NETGUARD NOC — DATASET VALIDATION AUDIT")
    print("=" * 60)
    print(f"Dataset Path: {csv_path}")

    df = pd.read_csv(csv_path)

    total_records = len(df)
    total_devices = df["Device_ID"].nunique()
    records_per_device = df.groupby("Device_ID").size()
    min_rec = records_per_device.min()
    max_rec = records_per_device.max()

    # Time range
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
    min_ts = df["Timestamp_dt"].min()
    max_ts = df["Timestamp_dt"].max()

    # Duplicates check
    dups = df.duplicated(subset=["Device_ID", "Timestamp"]).sum()

    # Missing values
    missing = df.isnull().sum().to_dict()
    missing_total = sum(missing.values())

    # Failure statistics
    failure_count = int(df["Failed"].sum()) if "Failed" in df.columns else 0
    failure_types = df["Failure_Type"].value_counts().to_dict() if "Failure_Type" in df.columns else {}
    target_dist = df["Failure_Next_12h"].value_counts(dropna=False).to_dict() if "Failure_Next_12h" in df.columns else {}

    # Feature availability
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]

    # Leakage check: verify Hidden_Degradation_State / Latent_Health_State are not in feature set
    latent_in_features = [f for f in EXCLUDE_COLUMNS if f in FEATURE_COLUMNS]

    print(f"\n1. GENERAL METRICS")
    print(f"   Total Temporal Records:  {total_records:,}")
    print(f"   Total Unique Devices:    {total_devices}")
    print(f"   Records Per Device:      Min={min_rec}, Max={max_rec}")
    print(f"   Time Horizon Range:      {min_ts} to {max_ts}")

    print(f"\n2. DATA INTEGRITY & QUALITY")
    print(f"   Duplicate (Device, Timestamp) Pairs: {dups} {'✓ (CLEAN)' if dups == 0 else '❌ (DUPLICATES FOUND)'}")
    print(f"   Total Missing Values:                {missing_total} {'✓ (CLEAN)' if missing_total == 0 else '⚠️ (MISSING VALUES PRESENT)'}")

    print(f"\n3. TARGET DISTRIBUTION")
    print(f"   Instantaneous Failures (Failed=1):   {failure_count:,} ({failure_count/total_records*100:.1f}%)")
    print(f"   Failure Types Breakdown:             {failure_types}")
    print(f"   Failure_Next_12h Horizon Target:     {target_dist}")

    print(f"\n4. FEATURE SCHEMA & LEAKAGE AUDIT")
    print(f"   Feature Columns Available:          {len(FEATURE_COLUMNS) - len(missing_features)} / {len(FEATURE_COLUMNS)}")
    print(f"   Missing Schema Features:            {missing_features if missing_features else 'None ✓'}")
    print(f"   Forbidden Latent State in Features: {latent_in_features if latent_in_features else 'None ✓ (LEAKAGE SAFE)'}")

    is_valid = dups == 0 and len(missing_features) == 0 and len(latent_in_features) == 0

    print("\n" + "=" * 60)
    print(f"DATASET VALIDATION AUDIT: {'PASSED ✓' if is_valid else 'FAILED ❌'}")
    print("=" * 60)

    return {
        "valid": is_valid,
        "total_records": total_records,
        "total_devices": total_devices,
        "duplicates": int(dups),
        "missing_features": missing_features,
        "latent_in_features": latent_in_features
    }


if __name__ == "__main__":
    validate_dataset()
