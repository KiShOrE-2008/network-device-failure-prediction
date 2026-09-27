"""REST API for advanced NetGuard intelligence."""
from flask import Blueprint,jsonify,request
from advanced_intelligence import analyze,forecast,heuristic_risk,security,root_cause,impact,correlate,runbook
advanced_bp=Blueprint("advanced_intelligence",__name__)
def body():
    x=request.get_json(silent=True);return x if isinstance(x,dict) else {}
@advanced_bp.get("/api/intelligence/capabilities")
def capabilities():return jsonify({"success":True,"modules":["MULTI_HORIZON_FORECAST","EXPLAINABLE_RISK","SECURITY_ANOMALY","INCIDENT_CORRELATION","ROOT_CAUSE_ANALYSIS","BLAST_RADIUS","FAILURE_SIMULATION","REMEDIATION_RUNBOOK"]})
@advanced_bp.post("/api/intelligence/analyze")
def analyze_route():
    p=body();return jsonify({"success":True,"device_analysis":[analyze(x) for x in p.get("devices",[]) if isinstance(x,dict)]})
@advanced_bp.post("/api/intelligence/root-cause")
def root_route():
    p=body();return jsonify({"success":True,**root_cause(p.get("devices",[]),p.get("links",[]),p.get("incidents",[]))})
@advanced_bp.post("/api/intelligence/correlate")
def correlate_route():
    p=body();return jsonify({"success":True,"groups":correlate(p.get("incidents",[]),p.get("links",[]))})
@advanced_bp.post("/api/intelligence/impact")
def impact_route():
    p=body();return jsonify({"success":True,**impact(p.get("root_device"),p.get("nodes",[]),p.get("links",[]))})
@advanced_bp.post("/api/intelligence/security")
def security_route():
    p=body();return jsonify({"success":True,"events":security(p.get("telemetry",p))})
@advanced_bp.post("/api/intelligence/simulate")
def simulate_route():
    p=body();base=dict(p.get("telemetry",{}))
    for k,v in p.get("changes",{}).items():
        try:base[k]=float(v)
        except (TypeError,ValueError):base[k]=v
    events=security(base)
    return jsonify({"success":True,"baseline_risk":heuristic_risk(p.get("telemetry",{})),"simulated_risk":heuristic_risk(base),"forecast":forecast(p.get("failure_probability"),base),"security_anomalies":events,"remediation":runbook(base,events,p.get("diagnosis","UNKNOWN"))})
@advanced_bp.post("/api/intelligence/remediation")
def remediation_route():
    p=body();return jsonify({"success":True,**runbook(p.get("telemetry",{}),p.get("security_events",[]),p.get("diagnosis","UNKNOWN"))})
