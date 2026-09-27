"""Advanced NOC intelligence engines for NetGuard."""
from collections import defaultdict, deque
HORIZONS=(1,6,12,24,48)
def num(v,d=0.0):
    try:return float(v)
    except (TypeError,ValueError):return d
def clamp(v):return round(max(0,min(100,float(v))),1)
def heuristic_risk(t):
    cpu=num(t.get("CPU_Usage",t.get("cpu")));mem=num(t.get("Memory_Usage",t.get("memory")));temp=num(t.get("Temperature",t.get("temperature")))
    err=min(num(t.get("Interface_Errors",t.get("errors"))),100);loss=min(num(t.get("Packet_Loss",t.get("packet_loss")))*10,100)
    trend=min(max(num(t.get("CPU_Trend"))+num(t.get("Temperature_Trend")),0),100)
    return clamp(cpu*.25+mem*.15+temp*.25+err*.15+loss*.10+trend*.10)
def forecast(probability,t):
    base=clamp(num(probability)*100 if probability is not None else heuristic_risk(t));trend=max(0,num(t.get("CPU_Trend"))+num(t.get("Temperature_Trend")))
    growth=min(18,trend*.35)
    return {"source":"ML" if probability is not None else "HEURISTIC","base_probability":round(num(probability),4) if probability is not None else None,"forecast_percent":{f"{h}h":clamp(base+(h/12)*(8+growth)) for h in HORIZONS}}
def explain(t):
    checks=[("Temperature",num(t.get("Temperature",t.get("temperature"))),70,1.25,"thermal stress"),("Interface Errors",num(t.get("Interface_Errors",t.get("errors"))),20,1.15,"interface instability"),("Packet Loss",num(t.get("Packet_Loss",t.get("packet_loss"))),2,1.1,"packet delivery degradation"),("Memory Usage",num(t.get("Memory_Usage",t.get("memory"))),80,.95,"memory pressure"),("CPU Usage",num(t.get("CPU_Usage",t.get("cpu"))),80,.9,"compute saturation"),("CPU Trend",num(t.get("CPU_Trend")),5,.75,"rising workload"),("Temperature Trend",num(t.get("Temperature_Trend")),3,.7,"thermal acceleration")]
    out=[]
    for name,v,th,w,reason in checks:
        if v>th:out.append({"feature":name,"value":round(v,2),"severity":clamp((v-th)/max(100-th,1)*100),"weight":w,"reason":reason})
    out.sort(key=lambda x:x["severity"]*x["weight"],reverse=True);total=sum(x["severity"]*x["weight"] for x in out) or 1
    for x in out:x["relative_contribution_pct"]=round(x["severity"]*x["weight"]/total*100,1)
    return out[:7]
def security(t):
    bw=num(t.get("Bandwidth_Usage",t.get("bandwidth")));con=num(t.get("Connection_Count",t.get("connections")));auth=num(t.get("Failed_Auth_Count",t.get("failed_auth")));logs=num(t.get("Log_Errors",t.get("log_errors")));loss=num(t.get("Packet_Loss",t.get("packet_loss")));o=[]
    if bw>=92:o.append({"type":"TRAFFIC_SPIKE","severity":"HIGH","score":clamp(bw),"evidence":f"Bandwidth {bw:.1f}%"})
    if con>=250:o.append({"type":"CONNECTION_SURGE","severity":"HIGH","score":clamp(con/4),"evidence":f"Connections {con:.0f}"})
    if auth>=10:o.append({"type":"AUTHENTICATION_ANOMALY","severity":"HIGH","score":clamp(auth*5),"evidence":f"Failed auth {auth:.0f}"})
    if logs>=25:o.append({"type":"LOG_ANOMALY","severity":"MEDIUM","score":clamp(logs*2),"evidence":f"Log errors {logs:.0f}"})
    if loss>=5:o.append({"type":"NETWORK_TRAFFIC_ANOMALY","severity":"MEDIUM","score":clamp(loss*10),"evidence":f"Packet loss {loss:.1f}%"})
    return o
def graph(nodes,links):
    g=defaultdict(set)
    for n in nodes:
        i=str(n.get("id",n.get("device_id","")))
        if i:g[i]
    for l in links:
        a,b=str(l.get("source","")),str(l.get("target",""))
        if a and b:g[a].add(b);g[b].add(a)
    return g
def root_cause(devices,links,incidents=()):
    affected=[str(d.get("device_id",d.get("id",""))) for d in devices if str(d.get("risk","LOW")).upper() in {"HIGH","CRITICAL"} or num(d.get("anomaly_score"))>=65]
    g=graph(devices,links);c=[]
    for n in g:
        overlap=len(g[n]&set(affected))+(1 if n in affected else 0)
        if overlap:c.append((overlap,len(g[n]),n))
    c.sort(reverse=True);candidate=c[0][2] if c else (affected[0] if affected else None)
    return {"root_candidate":candidate,"confidence":clamp(45+(c[0][0]*12 if c else 0)),"affected_devices":affected,"evidence":[f"{len(g.get(candidate,set()))} topology neighbors"] if candidate else [],"method":"topology-correlation","is_causal_proof":False}
def impact(root,nodes,links):
    g=graph(nodes,links)
    if not root or root not in g:return {"root_device":root,"affected_devices":[],"device_count":0,"impact_score":0}
    seen={root};q=deque([root])
    while q:
        for n in g[q.popleft()]:
            if n not in seen:seen.add(n);q.append(n)
    a=sorted(seen-{root});return {"root_device":root,"affected_devices":a,"device_count":len(a),"impact_score":clamp(len(a)*4)}
def correlate(incidents,links):
    g=graph([],links);groups=defaultdict(list)
    for i in incidents:
        d=str(i.get("device_id","UNKNOWN"));ns=sorted(g.get(d,set()));groups[min([d]+ns)].append(i)
    return [{"correlation_key":k,"incident_count":len(v),"devices":sorted({str(i.get("device_id")) for i in v}),"correlation_type":"topology-proximity" if len({str(i.get("device_id")) for i in v})>1 else "device-local"} for k,v in sorted(groups.items(),key=lambda x:len(x[1]),reverse=True)]
def runbook(t,events=(),diagnosis="UNKNOWN"):
    s=[]
    if num(t.get("Temperature",t.get("temperature")))>70:s+=["Inspect temperature and cooling state","Check fan, airflow and environmental telemetry"]
    if num(t.get("Interface_Errors",t.get("errors")))>20:s+=["Inspect interface error counters","Check cable/transceiver and peer interface"]
    if num(t.get("Packet_Loss",t.get("packet_loss")))>2:s+=["Measure end-to-end packet loss","Check upstream/downstream interfaces"]
    if diagnosis!="UNKNOWN":s.append(f"Validate diagnostic hypothesis: {diagnosis}")
    s += [f"Investigate security signal: {e.get('type')} ({e.get('evidence','')})" for e in events]
    return {"safe_to_auto_execute":False,"requires_operator_approval":True,"steps":s or ["Review telemetry timeline","Confirm device reachability","Escalate if degradation persists"]}
def analyze(d):
    t=d.get("telemetry",d);sec=security(t)
    return {"device_id":d.get("device_id","UNKNOWN"),"forecast":forecast(d.get("failure_probability"),t),"explainability":explain(t),"security_anomalies":sec,"remediation":runbook(t,sec,d.get("predicted_failure","UNKNOWN"))}
