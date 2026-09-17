import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Reproducibility
np.random.seed(42)

def generate_network_telemetry(num_devices=500, steps_per_device=100):
    """
    Generates realistic time-series network device telemetry with temporal degradation,
    multi-mode failure trajectories, rolling trend features, and syslog indicators.
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
        base_uptime = np.random.uniform(10, 800)
        
        # Degradation onset step (if failure mode is assigned)
        onset_step = np.random.randint(40, 75) if assigned_mode != "NONE" else 999
        
        device_configs.append({
            "device_id": dev_id,
            "device_type": dev_type,
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
    
    print(f"Generating temporal degradation telemetry for {num_devices} devices across {steps_per_device} timesteps...")
    
    for dev in device_configs:
        dev_id = dev["device_id"]
        dev_type = dev["device_type"]
        mode = dev["mode"]
        onset = dev["onset_step"]
        
        # State trackers
        curr_cpu = dev["base_cpu"]
        curr_mem = dev["base_mem"]
        curr_temp = dev["base_temp"]
        curr_iface_err = dev["base_iface_err"]
        curr_loss = dev["base_loss"]
        curr_bw = dev["base_bw"]
        curr_log_err = dev["base_log_err"]
        curr_uptime = dev["base_uptime"]

        for t in range(steps_per_device):
            timestamp = start_time + timedelta(hours=t)
            curr_uptime += 1/24.0 # 1 hour per step
            
            # Normal baseline fluctuation
            cpu = max(5.0, min(100.0, curr_cpu + np.random.normal(0, 2.0)))
            mem = max(10.0, min(100.0, curr_mem + np.random.normal(0, 1.0)))
            temp = max(20.0, min(100.0, curr_temp + np.random.normal(0, 0.8)))
            iface_err = max(0, int(curr_iface_err + np.random.poisson(lam=0.5)))
            loss = max(0.0, min(20.0, curr_loss + np.random.normal(0, 0.1)))
            bw = max(5.0, min(100.0, curr_bw + np.random.normal(0, 3.0)))
            log_err = max(0, int(curr_log_err + np.random.poisson(lam=0.2)))
            
            # If past onset timestep, apply degradation specific to failure mode
            if t >= onset:
                progress = (t - onset) / (steps_per_device - onset)
                
                if mode == "THERMAL":
                    temp += progress * 45.0 + np.random.uniform(0, 3)
                    cpu += progress * 30.0 + np.random.uniform(0, 5)
                    log_err += int(progress * 15)
                elif mode == "MEMORY":
                    mem += progress * 50.0 + np.random.uniform(0, 2)
                    log_err += int(progress * 20)
                elif mode == "INTERFACE":
                    iface_err += int(progress * 80 + np.random.poisson(lam=5))
                    loss += progress * 8.0 + np.random.uniform(0, 1.5)
                    log_err += int(progress * 10)
                elif mode == "CONGESTION":
                    bw += progress * 45.0 + np.random.uniform(0, 5)
                    loss += progress * 6.0 + np.random.uniform(0, 1.0)
                    cpu += progress * 25.0
                elif mode == "HARDWARE":
                    temp += progress * 20.0
                    iface_err += int(progress * 40)
                    log_err += int(progress * 25)
                    curr_uptime += 5.0 # Accelerated uptime proxy
            
            # Bound metrics
            cpu = round(min(100.0, max(5.0, cpu)), 2)
            mem = round(min(100.0, max(10.0, mem)), 2)
            temp = round(min(100.0, max(20.0, temp)), 2)
            iface_err = int(max(0, iface_err))
            loss = round(min(25.0, max(0.0, loss)), 2)
            bw = round(min(100.0, max(5.0, bw)), 2)
            log_err = int(max(0, log_err))
            uptime = round(curr_uptime, 2)
            
            # Evaluate failure condition
            # Calculate failure probability score
            score = (
                0.25 * (cpu / 100.0) +
                0.20 * (mem / 100.0) +
                0.20 * (temp / 90.0) +
                0.10 * (min(iface_err, 150) / 100.0) +
                0.10 * (loss / 10.0) +
                0.10 * (bw / 100.0) +
                0.05 * (min(log_err, 40) / 30.0)
            ) + np.random.normal(0, 0.02)
            
            failed = 1 if (score > 0.65 and mode != "NONE") else 0
            curr_failure_type = mode if failed == 1 else "NONE"
            
            # Generate syslog indicator message
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
                "Device_Type": dev_type,
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
                "Failure_Type": curr_failure_type,
                "Failed": failed
            })
            
            # Update state for next step
            curr_cpu, curr_mem, curr_temp = cpu, mem, temp
            curr_iface_err, curr_loss, curr_bw, curr_log_err = iface_err, loss, bw, log_err

    df = pd.DataFrame(records)
    
    # ---------------------------------------------------------------------------
    # Compute Rolling Trend & Spike Features per Device
    # ---------------------------------------------------------------------------
    print("Engineering temporal degradation trends & metric spike features...")
    df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Device_ID", "Timestamp_dt"]).reset_index(drop=True)
    
    # Grouped rolling calculations
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
    
    # Clean up temporary column
    df = df.drop(columns=["Timestamp_dt"])
    
    # Save datasets
    timeseries_path = "data/network_devices_timeseries.csv"
    main_path = "data/network_devices.csv"
    
    df.to_csv(timeseries_path, index=False)
    
    # Also save snapshot of latest readings for backward compatibility
    latest_df = df.groupby("Device_ID").last().reset_index()
    latest_df.to_csv(main_path, index=False)
    
    print("=" * 60)
    print("NetGuard NOC Time-Series Dataset Successfully Generated")
    print("=" * 60)
    print(f"Total Temporal Records: {len(df):,}")
    print(f"Total Unique Devices: {df['Device_ID'].nunique()}")
    print("\nFailure Label Distribution:")
    print(df["Failed"].value_counts())
    print("\nFailure Type Breakdown:")
    print(df["Failure_Type"].value_counts())
    print(f"\nTime-Series Dataset saved to: {timeseries_path}")
    print(f"Latest Snapshot Dataset saved to: {main_path}")

if __name__ == "__main__":
    generate_network_telemetry(num_devices=500, steps_per_device=100)