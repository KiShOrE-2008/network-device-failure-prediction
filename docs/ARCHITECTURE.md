# NetGuard NOC — System Architecture Specification

**System Name:** NetGuard NOC — Network Discovery & Fleet-Wide Predictive Failure Intelligence  
**Version:** 2.0.0  
**Architectural Pattern:** Modular Layered Backend with Vectorized ML Batch Pipeline, Relational Persistence, and Single Page Application (SPA) NOC Dashboard.

---

NetGuard NOC is designed for real-time monitoring, failure forecasting, anomaly detection, root cause diagnostics, and incident lifecycle management across enterprise network infrastructure.

```text
                    ┌─────────────────────┐
                    │ Authorized Network  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Network Discovery   │
                    │ ICMP / TCP / SNMP   │
                    │ REST / Simulation   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Device Inventory    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Telemetry Collector │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │ Feature Engineering         │
                 │                             │
                 │ Past-only rolling features  │
                 │ Trends                      │
                 │ Spikes                      │
                 └──────────────┬──────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
      ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
      │ Failure     │   │ Anomaly     │   │ Diagnostic   │
      │ Predictor   │   │ Detector    │   │ Engine       │
      │             │   │             │   │              │
      │ Next 12h    │   │ Isolation   │   │ Failure Mode │
      └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
             │                 │                  │
             └─────────────────┼──────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Fleet Intelligence  │
                    │                     │
                    │ Risk                │
                    │ Health              │
                    │ Ranking             │
                    │ Aggregation         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Incident Engine     │
                    │                     │
                    │ Active              │
                    │ Acknowledged        │
                    │ Resolved            │
                    └──────────┬──────────┘
                               │
               ┌───────────────┼────────────────┐
               ▼               ▼                ▼
        ┌────────────┐  ┌────────────┐  ┌─────────────┐
        │ REST APIs  │  │ Database   │  │ NOC         │
        │            │  │            │  │ Dashboard   │
        └────────────┘  └────────────┘  └─────────────┘
```

---

## 2. Core Subsystems & Components

### 2.1 Feature Engineering Subsystem (`src/feature_engineering.py`)
- **Authoritative Feature Schema:** Centralized schema (`CATEGORICAL_FEATURES`, `NUMERICAL_FEATURES`, `FEATURE_COLUMNS`).
- **Leakage Prevention:** Computes 5-step rolling averages using strictly past observations (`shift(1)`).
- **Target & Latent Exclusion:** Explicitly excludes `Hidden_Degradation_State`, `Latent_Health_State`, `Failed`, `Failure_Type`, and `Failure_Next_12h`.

### 2.2 Supervised Forecasting Engine (`src/train_model.py`)
- **Primary Target:** `Failure_Next_12h` (impending failure in next 12 hours).
- **Validation Strategy:** Deterministic chronological 80/20 train/test split per `Device_ID`.
- **Model Registry:** Serializes production models (`failure_model.pkl`, `diagnostic_model.pkl`, `anomaly_model.pkl`) and auto-generates `models/model_metadata.json`.

### 2.3 Fleet Prediction & Batch Inference (`src/fleet_predictor.py`)
- **Batch Processing:** Vectorized feature matrix processing across the entire fleet in a single pass.
- **Device Uniqueness Guarantee:** Exactly 1 prediction payload per unique `Device_ID`.
- **Fleet Ranking:** Sorts all predictions descending by failure probability.

### 2.4 Diagnostic & Health Engines (`src/intelligence/`, `src/health_engine.py`)
- **4-Tier Operational Risk:**
  - `LOW`: $< 0.30$
  - `MEDIUM`: $0.30 \le p < 0.65$
  - `HIGH`: $0.65 \le p < 0.85$
  - `CRITICAL`: $\ge 0.85$
- **Health Score:** Unified scale $[0, 100]$ integrating telemetry, rolling trends, spike penalties, failure risk, and anomaly index.

### 2.5 Incident Store & Deduplication (`src/history_store.py`)
- **Relational Tables:** `devices`, `telemetry`, `predictions`, `alerts`, `discovered_nodes`.
- **Incident Lifecycle:** `ACTIVE` $\to$ `ACKNOWLEDGED` $\to$ `RESOLVED`.
- **Deduplication:** Multiple predictions for a degraded device update the existing `ACTIVE`/`ACKNOWLEDGED` incident in place without creating duplicate alert records.

### 2.6 Discovery Subsystem (`src/discovery.py`)
- **Supported Modes:**
  - `SIMULATION`: Uses inventory dataset metadata.
  - `LAB`: Socket reachability probing on authorized local subnets.
  - `PRODUCTION`: Explicit interface adapter requiring authenticated SNMP/REST credentials.

---

## 3. Operational Flow

1. **Discovery / Ingestion:** Subnet discovery populates `devices` and `discovered_nodes`.
2. **Telemetry Aggregation:** Operational metrics stream into `telemetry`.
3. **Batch Forecasting:** `FleetPredictor` extracts latest observations, calculates rolling features, and predicts 12-hour failure probabilities.
4. **Anomaly & Diagnosis:** Anomalies evaluated via Isolation Forest; failure modes diagnosed via XGBoost multi-class classifier.
5. **Incident Management:** Predictions with `HIGH`/`CRITICAL` risk update or trigger deduplicated entries in `alerts`.
6. **API / NOC Serving:** Flask REST API exposes metrics to mission-control dashboard.
