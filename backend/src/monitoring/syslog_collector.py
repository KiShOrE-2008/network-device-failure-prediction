"""
src/monitoring/syslog_collector.py
----------------------------------
Syslog Intelligence Parser for NetGuard NOC.
Parses, categorizes, and extracts severity levels and features from network device syslog streams.
"""

import re
from typing import Dict, Any, List

SEVERITY_MAP = {
    0: "EMERGENCY",
    1: "ALERT",
    2: "CRITICAL",
    3: "ERROR",
    4: "WARNING",
    5: "NOTICE",
    6: "INFORMATIONAL",
    7: "DEBUG"
}

CATEGORY_PATTERNS = {
    "TEMPERATURE": [r"THERMAL", r"TEMP", r"OVERHEAT", r"FAN"],
    "MEMORY": [r"MALLOC", r"MEMORY", r"OOM", r"BUFFER"],
    "INTERFACE": [r"LINK", r"LINEPROTO", r"CRC", r"PHY", r"INTERFACE"],
    "CONGESTION": [r"QOS", r"DROP", r"SATURAT", r"QUEUE"],
    "HARDWARE": [r"BUS", r"VOLTAGE", r"PARITY", r"HARDWARE", r"CHASSIS"],
    "AUTHENTICATION": [r"AUTH", r"LOGIN", r"PASSWD", r"SECURITY", r"SSH"],
    "SYSTEM": [r"SYS", r"CONFIG", r"REBOOT", r"CRASH"]
}

def parse_syslog(log_line: str) -> Dict[str, Any]:
    """
    Parses a syslog text string into structured components: severity level, category, and clean text.
    Example line: '%ENVIRONMENT-3-THERMAL_CRITICAL: Temperature reached 86C'
    """
    if not log_line or log_line == "NORMAL_OPERATIONAL_STATE":
        return {
            "raw_log": log_line,
            "severity_code": 6,
            "severity_label": "INFORMATIONAL",
            "category": "SYSTEM",
            "is_critical": False
        }

    # Extract Cisco-style severity digit (e.g. -3- or -2-)
    match_sev = re.search(r"-([0-7])-", log_line)
    if match_sev:
        sev_code = int(match_sev.group(1))
    else:
        if "CRITICAL" in log_line or "FAIL" in log_line or "ERROR" in log_line:
            sev_code = 2
        elif "WARNING" in log_line or "WARN" in log_line:
            sev_code = 4
        else:
            sev_code = 5

    sev_label = SEVERITY_MAP.get(sev_code, "NOTICE")
    
    # Category matching
    upper_line = log_line.upper()
    detected_cat = "SYSTEM"
    for cat, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(p, upper_line) for p in patterns):
            detected_cat = cat
            break

    is_critical = sev_code <= 3

    return {
        "raw_log": log_line,
        "severity_code": sev_code,
        "severity_label": sev_label,
        "category": detected_cat,
        "is_critical": is_critical
    }

def analyze_syslog_stream(logs: List[str]) -> Dict[str, Any]:
    """
    Analyzes a buffer of syslog messages and produces category counts and critical incident flags.
    """
    parsed = [parse_syslog(l) for l in logs]
    critical_count = sum(1 for p in parsed if p["is_critical"])
    
    categories = {}
    for p in parsed:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "total_logs": len(logs),
        "critical_count": critical_count,
        "category_counts": categories,
        "parsed_logs": parsed
    }
