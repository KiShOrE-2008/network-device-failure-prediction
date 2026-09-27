import os,sys
SRC=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","src")
if SRC not in sys.path:sys.path.insert(0,SRC)
from advanced_intelligence import forecast,heuristic_risk,explain,security,root_cause,impact
def test_forecast_horizons():
 r=forecast(.4,{"CPU_Trend":8});assert r["source"]=="ML";assert list(r["forecast_percent"])==["1h","6h","12h","24h","48h"]
def test_heuristic_not_probability():
 assert 0<=heuristic_risk({"CPU_Usage":95,"Temperature":90,"Interface_Errors":40})<=100
def test_explainability():
 r=explain({"Temperature":90,"Interface_Errors":50,"Packet_Loss":7});assert r and r[0]["relative_contribution_pct"]>=r[-1]["relative_contribution_pct"]
def test_security_signals():
 r=security({"Bandwidth_Usage":98,"Failed_Auth_Count":20});assert {x["type"] for x in r}>={"TRAFFIC_SPIKE","AUTHENTICATION_ANOMALY"}
def test_rca_and_impact():
 n=[{"id":"CORE"},{"id":"SW1"},{"id":"SW2"}];l=[{"source":"CORE","target":"SW1"},{"source":"CORE","target":"SW2"}];d=[{"device_id":"CORE","risk":"CRITICAL"},{"device_id":"SW1","risk":"HIGH"},{"device_id":"SW2","risk":"HIGH"}]
 assert root_cause(d,l)["root_candidate"]=="CORE";assert impact("CORE",n,l)["device_count"]==2
