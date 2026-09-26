"""
src/generate_dataset.py
------------------------
Simulates realistic time-series network device telemetry driven by latent physical health state degradation.
Generates device metadata, degradation trends, spikes, syslog messages, and multi-mode failures.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Reproducibility
np.random.seed(42)

VENDORS = {
    "Router": [("Cisco", "ISR-4331"), ("Cisco", "ASR-1001-X"), ("Juniper", "MX204")],
    "Switch": [("Cisco", "Catalyst-9300"), ("Arista", "7050SX3"), ("Juniper", "EX4300")]
}

LOCATIONS = [
    ("Chennai DC-1", "Rack A01"),
    ("Chennai DC-1", "Rack B04"),
    ("Bangalore DC-2", "Rack C12"),
    ("Mumbai Edge", "Rack E03"),
    ("Hyderabad DC-1", "Rack H08")
]

FIRMWARE_VERSIONS = ["17.6.4", "17.9.2a", "21.4R3", "4.28.2F"]

def generate_network_telemetry(num_devices=500, steps_per_device=100):
    """
    Generates telemetry where failure is driven by a latent component health state (h_t),
    decoupling telemetry measurements from direct threshold formula calculation.
    """
    os.makedirs("data", exist_ok=True)
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    
    failure_modes = ["NONE", "THERMAL", "MEMORY", "INTERFACE", "CONGESTION", "HARDWARE"]
    failure_mode_probs = [0.65, 0.07, 0.07, 0.07, 0.07, 0.07]
    
    device_configs = []
    for i in range(1, num_devices + 1):
        dev_id = f"DEV-{i:05d}"
        dev_type = np.random.choice(["Router", "Switch"], p=[0.5, 0.5])
        assigned_mode = np.random.choice(failure_modes, p=failure_mode_probs)
        
        vendor, model = VENDORS[dev_type][np.random.choice(len(VENDORS[dev_type]))]
        location, rack = LOCATIONS[np.random.choice(len(LOCATIONS))]
        ip_addr = f"10.10.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}"
        hostname = f"{dev_type[:3].upper()}-{location[:3].upper()}-{i:03d}"
        firmware = np.random.choice(FIRMWARE_VERSIONS)
        base_uptime = np.random.uniform(10, 800)
        
        onset_step = np.random.randint(40, 75) if assigned_mode != "NONE" else 999
        
        device_configs.append({
            "device_id": dev_id,
            "hostname": hostname,
            "ip_address": ip_addr,
            "device_type": dev_type,
            "vendor": vendor,
            "model": model,
            "location": location,
            "rack": rack,
            "firmware": firmware,
            "mode": assigned_mode,
            "base_uptime": base_uptime,
            "onset_step": onset_step,
            "base_cpu": np.random.uniform(15, 35),
            "base_mem": np.random.uniform(25, 45),
            "base_temp": np.random.uniform(28, 42),
            "base_iface_err": np.random.poisson(lam=2),
            "base_loss": np.random.uniform(0.0, 0.5),
            "base_bw": np.random.uniform(15, 50),
            "base_log_err": np.random.poisson(lam=1)
        })

    records = []
    print(f"Generating latent-state telemetry for {num_devices} devices across {steps_per_device} timesteps...")
    
    for dev in device_configs:
        dev_id = dev["device_id"]
        dev_type = dev["device_type"]
        mode = dev["mode"]
        onset = dev["onset_step"]
        
        curr_cpu = dev["base_cpu"]
        curr_mem = dev["base_mem"]
        curr_temp = dev["base_temp"]
        curr_iface_err = dev["base_iface_err"]
        curr_loss = dev["base_loss"]
        curr_bw = dev["base_bw"]
        curr_log_err = dev["base_log_err"]
        curr_uptime = dev["base_uptime"]
        
        # Latent Physical Health State (100 = perfect, 0 = broken)
        latent_health = 100.0

        for t in range(steps_per_device):
            timestamp = start_time + timedelta(hours=t)
            curr_uptime += 1/24.0
            
            # Baseline physical fluctuation
            cpu = max(5.0, min(100.0, curr_cpu + np.random.normal(0, 2.0)))
            mem = max(10.0, min(100.0, curr_mem + np.random.normal(0, 1.0)))
            temp = max(20.0, min(100.0, curr_temp + np.random.normal(0, 0.8)))
            iface_err = max(0, int(curr_iface_err + np.random.poisson(lam=0.5)))
            loss = max(0.0, min(20.0, curr_loss + np.random.normal(0, 0.1)))
            bw = max(5.0, min(100.0, curr_bw + np.random.normal(0, 3.0)))
            log_err = max(0, int(curr_log_err + np.random.poisson(lam=0.2)))
            
            # Latent state degradation progression past onset
            if t >= onset:
                progress = (t - onset) / (steps_per_device - onset)
                # Physical component degrades
                latent_health -= progress * 2.5 + np.random.uniform(0.5, 1.5)
                latent_health = max(0.0, latent_health)
                
                # Telemetry reflects degradation noisy observations
                if mode == "THERMAL":
                    temp += (100.0 - latent_health) * 0.7 + np.random.uniform(0, 3)
                    cpu += (100.0 - latent_health) * 0.4 + np.random.uniform(0, 4)
                    log_err += int(progress * 12)
                elif mode == "MEMORY":
                    mem += (100.0 - latent_health) * 0.8 + np.random.uniform(0, 2)
                    log_err += int(progress * 18)
                elif mode == "INTERFACE":
                    iface_err += int((100.0 - latent_health) * 1.5 + np.random.poisson(lam=4))
                    loss += (100.0 - latent_health) * 0.15 + np.random.uniform(0, 1.2)
                    log_err += int(progress * 10)
                elif mode == "CONGESTION":
                    bw += (100.0 - latent_health) * 0.7 + np.random.uniform(0, 5)
                    loss += (100.0 - latent_health) * 0.12 + np.random.uniform(0, 1.0)
                    cpu += (100.0 - latent_health) * 0.3
                elif mode == "HARDWARE":
                    temp += (100.0 - latent_health) * 0.3
                    iface_err += int((100.0 - latent_health) * 0.8)
                    log_err += int(progress * 22)
            
            # Bound telemetry
            cpu = round(min(100.0, max(5.0, cpu)), 2)
            mem = round(min(100.0, max(10.0, mem)), 2)
            temp = round(min(100.0, max(20.0, temp)), 2)
            iface_err = int(max(0, iface_err))
            loss = round(min(25.0, max(0.0, loss)), 2)
            bw = round(min(100.0, max(5.0, bw)), 2)
            log_err = int(max(0, log_err))
            uptime = round(curr_uptime, 2)
            
            # Failure label triggers when latent physical health state drops below threshold (e.g. 35)
            failed = 1 if (latent_health < 35.0 and mode != "NONE") else 0
            curr_failure_type = mode if failed == 1 else "NONE"
            
            # Syslog generation
            if curr_failure_type == "THERMAL" or temp > 80:
                log_msg = f"%ENVIRONMENT-3-THERMAL_CRITICAL: Temperature reached {temp}C"
                syslog_crit = int(min(log_err, 10))
            elif curr_failure_type == "MEMORY" or mem > 90:
                log_msg = f"%SYS-2-MALLOCFAIL: Memory allocation failed, usage at {mem}%"
                syslog_crit = int(min(log_err, 10))
            elif curr_failure_type == "INTERFACE" or iface_err > 50:
                log_msg = f"%LINK-3-UPDOWN: Interface Gi0/1 flaps, errors count {iface_err}"
                syslog_crit = int(min(log_err, 8))
            elif curr_failure_type == "CONGESTION" or bw > 90:
                log_msg = f"%QOS-4-BUFFER_FULL: Interface buffer saturated, loss {loss}%"
                syslog_crit = int(min(log_err, 6))
            elif curr_failure_type == "HARDWARE":
                log_msg = "%HARDWARE-2-BUS_ERROR: Intermittent parity error detected on backplane"
                syslog_crit = int(min(log_err, 12))
            else:
                log_msg = "%SYS-5-CONFIG_I: Configured from console by admin" if t % 20 == 0 else "NORMAL_OPERATIONAL_STATE"
                syslog_crit = 0
            
            records.append({
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Device_ID": dev_id,
                "Hostname": dev["hostname"],
                "IP_Address": dev["ip_address"],
                "Device_Type": dev_type,
                "Vendor": dev["vendor"],
                "Model": dev["model"],
                "Location": dev["location"],
                "Rack": dev["rack"],
                "Firmware": dev["firmware"],
                "CPU_Usage": cpu,
                "Memory_Usage": mem,
                "Temperature": temp,
                "Uptime": uptime,
                "Interface_Errors": iface_err,
                "Packet_Loss": loss,
                "Bandwidth_Usage": bw,
                "Log_Errors": log_err,
                "Syslog_Critical_Count": syslog_crit,
                "Latest_Syslog": log_msg,
                "Latent_Health_State": round(latent_health, 2),
                "Failure_Type": curr_failure_type,
                "Failed": failed
            })
            
            curr_cpu, curr_mem, curr_temp = cpu, mem, temp
            curr_iface_err, curr_loss, curr_bw, curr_log_err = iface_err, loss, bw, log_err

    df = pd.DataFrame(records)
    
    # Engineer Trend & Spike Features
    print("Engineering temporal trends & metric spike indicators...")
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Device_ID", "Timestamp_dt"]).reset_index(drop=True)
    
    df["CPU_5step_avg"] = df.groupby("Device_ID")["CPU_Usage"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["CPU_Trend"] = df.groupby("Device_ID")["CPU_Usage"].transform(lambda x: x.diff(5).fillna(0))
    df["CPU_Spike"] = (df["CPU_Usage"] - df["CPU_5step_avg"] > 15.0).astype(int)
    
    df["Memory_Trend"] = df.groupby("Device_ID")["Memory_Usage"].transform(lambda x: x.diff(5).fillna(0))
    
    df["Temperature_5step_avg"] = df.groupby("Device_ID")["Temperature"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["Temperature_Trend"] = df.groupby("Device_ID")["Temperature"].transform(lambda x: x.diff(5).fillna(0))
    df["Temperature_Spike"] = (df["Temperature"] - df["Temperature_5step_avg"] > 8.0).astype(int)
    
    df["Error_Trend"] = df.groupby("Device_ID")["Interface_Errors"].transform(lambda x: x.diff(5).fillna(0))
    df["Error_Spike"] = (df["Interface_Errors"] - df["Error_Trend"] > 10).astype(int)
    
    df["PacketLoss_Trend"] = df.groupby("Device_ID")["Packet_Loss"].transform(lambda x: x.diff(5).fillna(0))
    
    df = df.drop(columns=["Timestamp_dt"])
    
    timeseries_path = "data/network_devices_timeseries.csv"
    main_path = "data/network_devices.csv"
    
    df.to_csv(timeseries_path, index=False)
    latest_df = df.groupby("Device_ID").last().reset_index()
    latest_df.to_csv(main_path, index=False)
    
    print("=" * 60)
    print("NetGuard NOC Telemetry Dataset Successfully Generated")
    print("=" * 60)
    print(f"Total Temporal Records: {len(df):,}")
    print(f"Total Unique Devices: {df['Device_ID'].nunique()}")
    print("\nFailure Distribution:")
    print(df["Failed"].value_counts())
    print("\nFailure Type Breakdown:")
    print(df["Failure_Type"].value_counts())
    print(f"\nSaved to {timeseries_path} and {main_path}")

if __name__ == "__main__":
    generate_network_telemetry(num_devices=500, steps_per_device=100)