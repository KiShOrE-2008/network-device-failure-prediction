# NetGuard NOC — Automated Test Suite & Correctness Verification

**Test Runner:** `pytest`  
**Execution Command:** `venv/bin/pytest backend/tests`

---

## Test Suite Coverage Overview

| Test Module | Coverage Domain | Key Verifications |
| :--- | :--- | :--- |
| `test_feature_engineering.py` | Feature Schema & Leakage | `Hidden_Degradation_State` exclusion, schema correctness, `shift(1)` past-only rolling window |
| `test_target_engineering.py` | `Failure_Next_12h` Target | Target does not use current `Failed(t)`, boundary rows $\to$ `NaN`, same device boundary isolation |
| `test_temporal_validation.py` | Temporal Split Correctness | Per-device 80/20 chronological split, $\max(t_{\text{train}}) < \min(t_{\text{test}})$ for every device |
| `test_fleet_predictor.py` | Fleet Predictor Engine | 500 unique device predictions, uniqueness, sorting by failure probability, 4-tier risk boundaries |
| `test_alert_deduplication.py` | Incident Lifecycle | Alert deduplication, in-place update for `ACTIVE`/`ACKNOWLEDGED` incidents, reopening resolved incidents |
| `test_discovery_modes.py` | Network Discovery | `SIMULATION`, `LAB`, and `PRODUCTION` modes, CIDR input validation |
| `test_api_endpoints.py` | REST API Integration | Fleet, device, alert, discovery, and topology endpoint contracts |
| `test_dataset.py`, `test_models.py`, `test_alerts.py`, etc. | Core Systems & Regression | Backward compatibility with legacy baseline test suite |

---

## Running Full Verification

To execute all unit and integration tests across the repository:

```bash
# Navigate to workspace root or backend directory
cd backend
../venv/bin/pytest tests
```
