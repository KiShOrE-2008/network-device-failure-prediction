"""
tests/test_model_contract.py
-----------------------------
Test suite for Model Registry contract & schema validation.
"""

import pytest
import os
import sys
import json

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(BACKEND_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_registry import ModelRegistry, ModelSchemaMismatchError
from feature_engineering import FEATURE_COLUMNS, ANOMALY_FEATURE_COLUMNS


def test_model_registry_valid_metadata(tmp_path):
    metadata_file = tmp_path / "model_metadata.json"
    valid_meta = {
        "target": "Failure_Next_12h",
        "feature_schema": {
            "all_features": FEATURE_COLUMNS
        }
    }
    metadata_file.write_text(json.dumps(valid_meta))

    registry = ModelRegistry(metadata_path=str(metadata_file))
    meta = registry.load_and_validate_metadata()
    assert meta["target"] == "Failure_Next_12h"


def test_model_registry_invalid_target(tmp_path):
    metadata_file = tmp_path / "model_metadata.json"
    invalid_meta = {
        "target": "Wrong_Target",
        "feature_schema": {
            "all_features": FEATURE_COLUMNS
        }
    }
    metadata_file.write_text(json.dumps(invalid_meta))

    registry = ModelRegistry(metadata_path=str(metadata_file))
    with pytest.raises(ModelSchemaMismatchError, match="Invalid model target"):
        registry.load_and_validate_metadata()


def test_model_registry_missing_features(tmp_path):
    metadata_file = tmp_path / "model_metadata.json"
    invalid_meta = {
        "target": "Failure_Next_12h",
        "feature_schema": {
            "all_features": ["CPU_Usage"]  # Incomplete feature list
        }
    }
    metadata_file.write_text(json.dumps(invalid_meta))

    registry = ModelRegistry(metadata_path=str(metadata_file))
    with pytest.raises(ModelSchemaMismatchError, match="Missing required feature columns"):
        registry.load_and_validate_metadata()
