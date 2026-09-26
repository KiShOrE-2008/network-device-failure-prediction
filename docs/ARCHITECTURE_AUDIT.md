# NetGuard NOC — Architecture Audit Report

**Date:** 2026-09-26  
**Auditor:** Lead Software Architect & Senior ML/Backend Engineer  
**Project:** NetGuard NOC — Network Discovery & Fleet-Wide Predictive Failure Intelligence

---

## 1. Executive Summary

This architecture audit evaluates the current implementation of the `network-device-failure-prediction` repository to outline necessary corrections and structural enhancements before proceeding with Phase 1-10 implementation. The system is being upgraded from a device-level predictive-maintenance demo into a production-style NOC fleet intelligence platform.

---

## 2. Current Architecture Overview

- **Directory Layout:**
  - `frontend/`: SPA static files (`index.html`, `style.css`, `app.js`, `app_additions.js`, `logo.png`).
  - `backend/`:
    - `data/`: Dataset CSVs (`network_devices.csv`, `network_devices_timeseries.csv`, `topology.csv`, `failure_events.csv`) and SQLite DB (`network_noc.db`).
    - `models/`: Trained model binaries (`failure_model.pkl`, `diagnostic_model.pkl`, `anomaly_model.pkl`).
    - `outputs/`: Visualization outputs and model evaluation reports.
    - `tests/`: Automated pytest unit and API test suite.
    - `src/`: Core Python modules, including:
      - `feature_engineering.py`: Temporal features and schema definition.
      - `generate_dataset.py`: Latent physical state dataset simulator.
      - `train_model.py`: Multi-model training and evaluation script.
      - `fleet_predictor.py`: Vectorized batch predictor and fleet ranker.
      - `health_engine.py`: 4-tier risk classification and health scoring.
      - `history_store.py`: SQLite persistence and alert deduplication.
      - `discovery.py`: Multi-mode CIDR discovery engine.
      - `web_app.py`: Flask web application server registering modular blueprints (`api/`).

---

## 3. Identified Problems & Risk Areas

### 3.1 Temporal Validation Flaw (Forecasting Leakage)
- **Problem:** `train_model.py` currently uses `GroupShuffleSplit` by `Device_ID` to split training (80%) and testing (20%) devices randomly.
- **Impact:** For time-series forecasting target `Failure_Next_12h` (predicting if a device fails within the next 12 hours given telemetry at time $t$), splitting by device does not enforce chronological boundaries. The model is tested on arbitrary device timelines instead of evaluating future time steps given past data.
- **Fix:** Implement a deterministic temporal split per device: first 80% chronological observations per device $\to$ TRAIN, remaining 20% $\to$ TEST. Guarantee $\max(T_{\text{train}}) < \min(T_{\text{test}})$ for every device.

### 3.2 Target Formulation (`Failure_Next_12h`)
- **Problem:** `Failure_Next_12h` target generation needs explicit formalization across all dataset generators and verification tests.
- **Impact:** Target must indicate if `Failed == 1` occurs at any step $t+1 \dots t+12$ for the *same device*. The last 12 observations for each device lack a full 12-hour future window and must be excluded from supervised training/evaluation to prevent incomplete boundary labeling.
- **Fix:** Update `generate_dataset.py` to calculate exact 12-hour lookahead targets per device and strip incomplete boundary rows during training. Add `test_future_target_does_not_use_current_failure`.

### 3.3 Potential Feature Leakage & Schema Consistency
- **Problem:** `Latent_Health_State` / `Hidden_Degradation_State` is a simulation latent variable.
- **Impact:** If included in model training, features leak future ground-truth states.
- **Fix:** Enforce `feature_engineering.py` as the single authoritative schema source. Exclude `Latent_Health_State`, `Hidden_Degradation_State`, `Failed`, `Failure_Type`, `Failure_Next_12h` from feature sets. Add `test_hidden_degradation_not_in_features`.

### 3.4 Diagnostic Model Validation Overfitting
- **Problem:** `train_model.py` evaluates the multi-class diagnostic classifier on its training set (`y_diag_pred = diagnostic_pipeline.predict(X_diag)`), reporting misleading 100% training accuracy.
- **Impact:** True out-of-sample diagnostic precision and weighted F1 score are unmeasured.
- **Fix:** Apply chronological train/test splitting to failure observations (`Failed == 1`), training on earlier failures and evaluating on later failure instances.

### 3.5 Duplicate API Routes in `web_app.py`
- **Problem:** `web_app.py` registers modular blueprints (`api/alerts.py`, `api/devices.py`, etc.) but also defines duplicate route handlers (`/api/alerts`, `/api/alerts/<id>/acknowledge`, etc.) directly on the Flask app object.
- **Impact:** Route shadowing and maintenance confusion.
- **Fix:** Remove inline duplicate route handlers from `web_app.py` and delegate all API routes cleanly to registered Flask blueprints under `src/api/`.

### 3.6 Incident Deduplication & Lifecycle
- **Problem:** Repeated predictions on a degraded device must update the existing ACTIVE/ACKNOWLEDGED incident rather than creating duplicate alert records.
- **Fix:** Audit `history_store.py` to ensure active incidents are updated in place with latest timestamp and severity, and resolved incidents re-open if a new critical risk recurs.

### 3.7 Discovery Mode Separation
- **Problem:** Network discovery must explicitly separate `SIMULATION`, `LAB`, and `PRODUCTION` capability modes without pretending simulated metadata is real SNMP production data.
- **Fix:** Implement explicit adapters for ICMP/TCP reachability probing in `LAB` mode, structured simulation inventory in `SIMULATION` mode, and clear production protocol interfaces/placeholders in `PRODUCTION` mode.

---

## 4. Planned Changes by Implementation Phase

| Phase | Description | Affected Files |
| :--- | :--- | :--- |
| **Phase 1** | Feature Engineering & Schema Standardization | `backend/src/feature_engineering.py`, `backend/tests/test_feature_engineering.py` |
| **Phase 2** | Target Correctness (`Failure_Next_12h`) | `backend/src/generate_dataset.py`, `backend/tests/test_dataset.py` |
| **Phase 3** | Temporal Split & Model Training Pipeline | `backend/src/train_model.py`, `backend/models/model_metadata.json` |
| **Phase 4** | Fleet Predictor & Vectorized Inference | `backend/src/fleet_predictor.py`, `backend/tests/test_fleet.py` |
| **Phase 5** | Database Inventory & Incident Lifecycle | `backend/src/history_store.py`, `backend/tests/test_alerts.py` |
| **Phase 6** | Discovery Engine Modes (SIMULATION/LAB/PRODUCTION)| `backend/src/discovery.py`, `backend/tests/test_discovery.py` |
| **Phase 7** | API Blueprint Cleanup & Route Deduplication | `backend/src/web_app.py`, `backend/src/api/*.py`, `backend/tests/test_api.py` |
| **Phase 8** | Integration & Final CLI Execution (`-m src.fleet_predictor`) | `backend/src/fleet_predictor.py`, `backend/app.py` |
| **Phase 9** | Frontend Audit & Contract Verification | `frontend/app.js`, `frontend/index.html` |
| **Phase 10**| Final Verification, Documentation & Reports | `docs/*.md`, `outputs/reports/` |

---

## 5. Risk Assessment

- **Data Integrity Risk:** Re-generating the dataset must maintain backward compatibility with existing tests.
- **Model Migration Risk:** Updating model signatures or feature inputs must update both training pipeline and batch predictor simultaneously.
- **API Breaking Changes:** Updating API payload keys must be validated against `frontend/app.js` to ensure the NOC dashboard renders without errors.

---

*Audit Complete. Proceeding with Phase 1.*
