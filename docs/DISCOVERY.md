# NetGuard NOC — Discovery Subsystem Specification

---

## Operational Discovery Modes

The Network Discovery engine (`src/discovery.py`) operates across 3 explicitly separated capability modes:

### 1. `SIMULATION` Mode
- **Purpose:** Engineering, demonstration, and dataset benchmarking.
- **Behavior:** Loads device metadata directly from benchmark dataset inventory (`data/network_devices.csv`).
- **Discovery Protocol Tag:** `SIMULATED_INVENTORY`

### 2. `LAB` Mode
- **Purpose:** Limited reachability probing on local lab networks.
- **Behavior:** Performs TCP socket reachability checks across authorized IP host ranges on common management ports (`22`, `80`, `443`).
- **Metadata Note:** Reachability probing alone **does not** identify hardware vendor, model, or firmware. Discovered nodes are explicitly tagged with `PROBED_REACHABLE` vendor status.
- **Discovery Protocol Tag:** `ICMP_TCP_SOCKET_PROBE`

### 3. `PRODUCTION` Mode
- **Purpose:** Production network monitoring across enterprise subnets.
- **Behavior:** Requires authenticated management credentials (`SNMP_COMMUNITY`, `NETCONF_USER`).
- **Security Constraint:** Does **not** perform unauthenticated or unauthorized probing. If credentials are missing, returns explicit adapter status requesting SNMP v2c/v3 authentication.
- **Discovery Protocol Tag:** `SNMP_V3_REST_API`

---

## Input Validation & Security

- **CIDR Validation:** All CIDR parameters are validated using `ipaddress.ip_network(strict=False)`. Invalid subnet strings raise an explicit `ValueError`.
- **Credential Storage:** Secrets and community strings are fetched strictly from environment variables and never hardcoded in source control.
