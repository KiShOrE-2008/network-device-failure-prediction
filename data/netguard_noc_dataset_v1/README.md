# NetGuard NOC Dataset v1

Synthetic network telemetry dataset designed for network-device failure prediction,
diagnosis, anomaly detection, and NOC visualization.

## Files

- `network_devices_timeseries.csv` — 50,000 hourly telemetry records
- `network_devices.csv` — 500 device inventory records
- `failure_events.csv` — event-level failure ground truth
- `topology.csv` — device relationships for the SVG topology view

## Scale

- Devices: 500
- Timesteps/device: 100
- Telemetry rows: 50,000
- Time resolution: 1 hour
- Random seed: 42

## Important targets

- `Failed`: current failure state.
- `Failure_Type`: current failure mode when failed.
- `Failure_Next_12h`: whether a failure occurs during the next 12 hours.
- `Hidden_Degradation_State`: simulation-only latent state. **Do not use this as an ML feature.**

## Failure modes

THERMAL, MEMORY, INTERFACE, CONGESTION, HARDWARE.

## Dataset design

The generator uses a latent degradation process and then manifests degradation
through observable telemetry. This avoids defining the target as a direct weighted
formula of the same features used for prediction.

Temporal features use past-only rolling baselines to reduce current-observation leakage.

## Research note

This is a synthetic benchmark. It is intended for engineering/demo/testing and should
not be presented as evidence of real-world model performance. Public real-world/
research datasets can be used as external validation references.
