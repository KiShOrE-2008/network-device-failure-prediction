# NetGuard NOC — ML Pipeline & Temporal Forecasting Specification

---

## 1. Primary Forecasting Target (`Failure_Next_12h`)

The primary ML model forecasts impending network device failure over a **12-hour forward-looking horizon**:

$$\text{Failure\_Next\_12h}(t) = \begin{cases} 1.0 & \text{if } \text{Failed}(t') = 1 \text{ for any } t' \in [t+1, t+12] \text{ on the same device} \\ 0.0 & \text{otherwise} \end{cases}$$

### Strict Target Rules:
1. Does **NOT** include current $\text{Failed}(t)$ or current $\text{Failure\_Type}(t)$.
2. Does **NOT** include latent state variables ($\text{Hidden\_Degradation\_State}$ / $\text{Latent\_Health\_State}$).
3. Boundary horizon rows (the final 12 observations for each device) are assigned `NaN` and excluded from training and evaluation.

---

## 2. Temporal Validation Strategy

To prevent data leakage in time-series forecasting, validation is performed using a **deterministic chronological split per device**:

- For every `Device_ID`:
  - First $80\%$ of observations ordered by timestamp $\to$ **TRAIN**
  - Final $20\%$ of observations ordered by timestamp $\to$ **TEST**
- Enforces:

$$\max(t_{\text{train}}) < \min(t_{\text{test}})$$

for every individual device in the dataset.

---

## 3. Feature Schema & Leakage Prevention

All model features are centralized in `src/feature_engineering.py`:

- **Categorical Features:** `Device_Type`
- **Numerical Features:** `CPU_Usage`, `Memory_Usage`, `Temperature`, `Interface_Errors`, `Packet_Loss`, `Bandwidth_Usage`, `Uptime`, `Log_Errors`, `Syslog_Critical_Count`, `CPU_5step_avg`, `CPU_Trend`, `CPU_Spike`, `Memory_5step_avg`, `Memory_Trend`, `Temperature_5step_avg`, `Temperature_Trend`, `Temperature_Spike`, `Error_5step_avg`, `Error_Trend`, `Error_Spike`, `PacketLoss_5step_avg`, `PacketLoss_Trend`
- **Past-Only Rolling Transformations:** Rolling means and trend diffs use `shift(1)` to ensure no future time-step information leaks into features.

---

## 4. Multi-Model Architecture

| Model Role | Algorithm | Target Variable | Evaluation Set |
| :--- | :--- | :--- | :--- |
| **Failure Forecaster** | XGBoost Classifier (scale_pos_weight tuned) | `Failure_Next_12h` | Chronological Test Set (9,000 rows) |
| **Diagnostic Engine** | XGBoost Multi-Class Classifier | `Failure_Type` | Test Failures Set (1,056 rows) |
| **Anomaly Detector** | Isolation Forest (150 estimators) | Operational Baseline | Unsupervised Telemetry Matrix |

---

## 5. Model Metadata Generation

Model metadata is automatically generated during `train_model.py` execution and written to `models/model_metadata.json`. It includes:
- Training timestamp
- Dataset name and version
- Total records and device counts
- Target variable definition
- Feature schema and categorical/numerical columns
- Selected model name and validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
