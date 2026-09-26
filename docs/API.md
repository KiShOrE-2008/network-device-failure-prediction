# NetGuard NOC — REST API Reference Documentation

**Base URL:** `http://localhost:5000`  
**Content-Type:** `application/json`

---

## 1. Fleet Endpoints

### `GET /api/fleet/predictions`
Returns predictions for all devices in the network fleet, sorted descending by failure risk.

**Response `200 OK`:**
```json
{
  "success": true,
  "fleet": {
    "total_devices": 500,
    "risk_summary": {
      "low": 480,
      "medium": 12,
      "high": 5,
      "critical": 3
    },
    "network_health_score": 86.5,
    "average_failure_probability": 4.2
  },
  "predictions": [
    {
      "device_id": "DEV-00042",
      "hostname": "ROU-CHE-042",
      "ip_address": "10.10.4.12",
      "device_type": "Router",
      "vendor": "Cisco",
      "model": "ISR-4331",
      "location": "Chennai DC-1",
      "rack": "Rack A01",
      "firmware": "17.6.4",
      "timestamp": "2026-01-05T00:00:00",
      "failure_probability": 0.912,
      "failure_probability_pct": 91.2,
      "risk": "CRITICAL",
      "predicted_failure": "THERMAL",
      "diagnostic_confidence": 0.95,
      "anomaly_score": 82.5,
      "health_score": 14.2,
      "current_failed": 0,
      "recommended_actions": [
        "Inspect cooling fans, clean chassis vents, reduce ambient heat."
      ],
      "telemetry": {
        "cpu": 88.5,
        "memory": 72.0,
        "temperature": 84.2,
        "errors": 12,
        "packet_loss": 0.5,
        "bandwidth": 65.0,
        "log_errors": 5
      }
    }
  ]
}
```

### `GET /api/fleet/stats`
Returns high-level NOC fleet statistics and failure mode breakdown.

### `GET /api/fleet/health`
Returns overall network health score and operational status (`OPERATIONAL`, `DEGRADED`, `CRITICAL`).

### `POST /api/fleet/predict`
Triggers batch inference over all devices and logs predictions to the database.

---

## 2. Device Endpoints

### `GET /api/devices`
Returns list of all devices in the fleet inventory with risk status.

### `GET /api/devices/<device_id>`
Returns comprehensive deep-dive details and historical prediction logs for a specific device.

---

## 3. Incident / Alert Endpoints

### `GET /api/alerts`
Returns active and acknowledged NOC incident alerts.

**Response `200 OK`:**
```json
{
  "success": true,
  "count": 1,
  "alerts": [
    {
      "id": 1,
      "device_id": "DEV-00042",
      "timestamp": "2026-01-05T00:00:00",
      "severity": "CRITICAL",
      "title": "CRITICAL Incident on DEV-00042",
      "message": "Risk: CRITICAL (91.2%), Mode: THERMAL, Anomaly Index: 82.5%",
      "status": "ACTIVE",
      "failure_type": "THERMAL",
      "probability": 0.912,
      "anomaly_score": 82.5
    }
  ]
}
```

### `POST /api/alerts/<id>/acknowledge`
Marks an active incident alert as `ACKNOWLEDGED`.

### `POST /api/alerts/<id>/resolve`
Marks an active or acknowledged incident alert as `RESOLVED`.

---

## 4. Discovery & Topology Endpoints

### `POST /api/discovery/scan`
Triggers subnet IP range scanning & registers discovered endpoints.

**Request Payload:**
```json
{
  "cidr": "10.1.0.0/24",
  "mode": "SIMULATION" // "SIMULATION" | "LAB" | "PRODUCTION"
}
```

### `GET /api/discovery/nodes`
Returns list of all discovered network endpoints.

### `GET /api/topology`
Returns parent-child network topology nodes and links for SVG rendering.
