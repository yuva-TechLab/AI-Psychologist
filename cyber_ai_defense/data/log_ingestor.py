

import re, json, time, queue, random, threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ING_DIR  = BASE_DIR / "ingestion"
ING_DIR.mkdir(parents=True, exist_ok=True)

# ── Data classes ──────────────────────────────────────────────

@dataclass
class LogEvent:
    raw_line: str; timestamp: str; src_ip: str; dst_ip: str
    event_type: str; details: dict = field(default_factory=dict); log_format: str = "unknown"
    def __str__(self): return f"[{self.timestamp}] {self.src_ip} -> {self.dst_ip} | {self.event_type}"

@dataclass
class StagedEvent:
    event: LogEvent; stage: str; confidence: float; rule_matched: str
    def __str__(self):
        return (f"[{self.event.timestamp}] {self.event.src_ip:<16} "
                f"-> {self.stage:<25} ({self.confidence:.0%}) [{self.rule_matched}]")

# ── Parser ────────────────────────────────────────────────────

LOG_PATTERNS = {
    "syslog_ssh_fail":   (re.compile(r"(?P<ts>\w{3}\s+\d+\s+[\d:]+).*sshd.*Failed password.*from\s+(?P<src>[\d.]+).*port\s+(?P<sport>\d+)"), "syslog", "ssh_failed"),
    "syslog_ssh_accept": (re.compile(r"(?P<ts>\w{3}\s+\d+\s+[\d:]+).*sshd.*Accepted (?P<method>\w+).*from\s+(?P<src>[\d.]+)"), "syslog", "ssh_accepted"),
    "syslog_sudo":       (re.compile(r"(?P<ts>\w{3}\s+\d+\s+[\d:]+).*sudo.*(?P<user>\w+).*COMMAND=(?P<cmd>.+)"), "syslog", "sudo_command"),
    "apache":            (re.compile(r'(?P<src>[\d.]+).*\[(?P<ts>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<path>[^\s]+).*"\s+(?P<status>\d+)'), "apache", "http_request"),
    "zeek_conn":         (re.compile(r"(?P<ts>[\d.]+)\s+\S+\s+(?P<src>[\d.]+)\s+(?P<sport>\d+)\s+(?P<dst>[\d.]+)\s+(?P<dport>\d+)\s+(?P<proto>\w+)\s+\S+\s+(?P<bytes>\d+)"), "zeek", "network_conn"),
    "windows_event":     (re.compile(r"(?P<ts>[\d/\s:]+),(?P<eid>\d+),(?P<level>\w+),(?P<src>[^,]+),(?P<msg>[^\n]+)"), "windows", "windows_event"),
}

SUSPICIOUS_PATHS = ["/etc/passwd","/etc/shadow","/.env","/admin","/wp-admin","/../","/cgi-bin/","phpinfo","cmd=","exec(","eval(","UNION SELECT","../","%2e%2e","sqlmap","nikto"]
SUSPICIOUS_PORTS = {22,23,445,3389,4444,1337,9999,5900,6666,31337}

class LogLineParser:
    def parse(self, line: str, default_src="0.0.0.0"):
        line = line.strip()
        if not line or line.startswith("#"): return None
        if line.startswith("{"): return self._parse_json(line)
        for name,(pat,fmt,et) in LOG_PATTERNS.items():
            m = pat.search(line)
            if m:
                gd = m.groupdict()
                details = {k:v for k,v in gd.items() if k not in ("ts","src","dst")}
                if et == "http_request":
                    path = details.get("path","")
                    details["suspicious"] = any(s in path for s in SUSPICIOUS_PATHS)
                    details["status_code"] = int(details.get("status",200))
                    details["bytes"] = int(details.get("bytes",0) if "bytes" in details else 0)
                return LogEvent(line, gd.get("ts",datetime.utcnow().isoformat()),
                                gd.get("src","0.0.0.0"), gd.get("dst","target"),
                                et, details, fmt)
        ip = re.search(r'\b(\d{1,3}(?:\.\d{1,3}){3})\b', line)
        return LogEvent(line, datetime.utcnow().isoformat(),
                        ip.group(1) if ip else default_src, "target",
                        "generic", {"raw": line[:200]}, "generic")

    def _parse_json(self, line):
        try:
            d = json.loads(line)
            return LogEvent(line, d.get("timestamp",datetime.utcnow().isoformat()),
                            d.get("src_ip","0.0.0.0"), d.get("dst_ip","target"),
                            d.get("event_type","json_event"),
                            {k:v for k,v in d.items() if k not in ("timestamp","src_ip","dst_ip","event_type")},
                            "json")
        except: return None

# ── Classifier ────────────────────────────────────────────────

RULES = [
    ("port_scan",     lambda e: True,                                                              "Reconnaissance",       0.95, "port_scan"),
    ("http_request",  lambda e: e.details.get("status_code") in (404,403) and e.details.get("suspicious"), "Reconnaissance", 0.80, "web_enum"),
    ("http_request",  lambda e: e.details.get("status_code") == 404,                              "Reconnaissance",       0.60, "web_404"),
    ("network_conn",  lambda e: int(e.details.get("dport",0)) in SUSPICIOUS_PORTS,                "Reconnaissance",       0.70, "suspicious_port"),
    ("ssh_failed",    lambda e: True,                                                              "Credential Access",    0.90, "ssh_brute"),
    ("windows_event", lambda e: e.details.get("eid") in ("4625","4771"),                          "Credential Access",    0.92, "win_failed_logon"),
    ("http_request",  lambda e: "admin" in e.details.get("path","") and e.details.get("status_code") in (401,403), "Credential Access", 0.82, "web_admin_brute"),
    ("http_request",  lambda e: e.details.get("suspicious") and e.details.get("status_code")==200,"Exploitation",         0.88, "web_exploit_success"),
    ("windows_event", lambda e: e.details.get("eid") in ("4688","4689"),                          "Exploitation",         0.75, "suspicious_process"),
    ("sudo_command",  lambda e: any(k in e.details.get("cmd","") for k in ("crontab","rc.local","systemctl enable","schtasks","reg add")), "Persistence", 0.88, "persistence_cmd"),
    ("windows_event", lambda e: e.details.get("eid") in ("4698","7045"),                          "Persistence",          0.90, "scheduled_task"),
    ("sudo_command",  lambda e: True,                                                              "Privilege Escalation", 0.82, "sudo_use"),
    ("windows_event", lambda e: e.details.get("eid") in ("4672","4673"),                          "Privilege Escalation", 0.86, "priv_account"),
    ("ssh_accepted",  lambda e: True,                                                              "Lateral Movement",     0.75, "ssh_accepted"),
    ("network_conn",  lambda e: int(e.details.get("dport",0)) in (445,135,139),                   "Lateral Movement",     0.85, "smb_lateral"),
    ("windows_event", lambda e: e.details.get("eid") in ("4648","4624"),                          "Lateral Movement",     0.72, "explicit_logon"),
    ("network_conn",  lambda e: int(e.details.get("bytes",0))>500000 and int(e.details.get("dport",0)) in (80,443,8080), "Command & Control", 0.78, "https_beacon"),
    ("network_conn",  lambda e: int(e.details.get("dport",0)) in (4444,1337,9999),                "Command & Control",    0.95, "c2_port"),
    ("network_conn",  lambda e: int(e.details.get("bytes",0))>10_000_000,                         "Data Exfiltration",    0.84, "large_transfer"),
    ("windows_event", lambda e: e.details.get("eid") in ("1102","104"),                           "Impact",               0.90, "log_cleared"),
    ("generic",       lambda e: any(k in e.details.get("raw","").lower() for k in ("ransom","encrypt","wiper","deleted")), "Impact", 0.88, "ransom_keyword"),
]

DEFAULTS = {
    "ssh_failed":"Credential Access","ssh_accepted":"Lateral Movement",
    "sudo_command":"Privilege Escalation","port_scan":"Reconnaissance",
    "http_request":"Reconnaissance","network_conn":"Reconnaissance",
    "windows_event":"Discovery","json_event":"Discovery","generic":"Discovery",
}

class StageClassifier:
    def classify(self, event: LogEvent) -> StagedEvent:
        for et,cond,stage,conf,rule in RULES:
            if event.event_type == et:
                try:
                    if cond(event): return StagedEvent(event,stage,conf,rule)
                except: continue
        stage = DEFAULTS.get(event.event_type,"Discovery")
        return StagedEvent(event, stage, 0.40, "fallback")

# ── Session tracker ───────────────────────────────────────────

class SessionTracker:
    def __init__(self, window=10, ttl=300):
        self._sessions = defaultdict(lambda: deque(maxlen=window))
        self._last_seen = {}; self._lock = threading.Lock(); self._ttl = ttl
    def ingest(self, staged: StagedEvent) -> list:
        ip,now = staged.event.src_ip, time.time()
        with self._lock:
            if ip in self._last_seen and (now-self._last_seen[ip]) > self._ttl:
                self._sessions[ip].clear()
            sess = self._sessions[ip]
            if not sess or sess[-1] != staged.stage: sess.append(staged.stage)
            self._last_seen[ip] = now
            return list(sess)
    def active_sessions(self):
        now = time.time()
        with self._lock:
            return {ip:list(s) for ip,s in self._sessions.items()
                    if s and (now-self._last_seen.get(ip,0))<self._ttl}

# ── Log tailer ────────────────────────────────────────────────

SYNTHETIC_LOGS = [
    "Mar 13 02:00:01 fw nmap: Nmap scan from 192.168.1.50 to 10.0.0.100",
    '192.168.1.50 - - [13/Mar/2026:02:00:10 +0000] "GET /admin HTTP/1.1" 404 512',
    '192.168.1.50 - - [13/Mar/2026:02:00:11 +0000] "GET /.env HTTP/1.1" 404 0',
    '192.168.1.50 - - [13/Mar/2026:02:00:12 +0000] "GET /wp-admin HTTP/1.1" 403 1024',
    "Mar 13 02:01:00 web sshd[1234]: Failed password for root from 192.168.1.50 port 51234",
    "Mar 13 02:01:01 web sshd[1234]: Failed password for admin from 192.168.1.50 port 51235",
    "Mar 13 02:01:02 web sshd[1234]: Failed password for ubuntu from 192.168.1.50 port 51236",
    '192.168.1.50 - - [13/Mar/2026:02:02:00 +0000] "GET /cgi-bin/test.sh?cmd=id HTTP/1.1" 200 48',
    '192.168.1.50 - - [13/Mar/2026:02:02:05 +0000] "POST /login?UNION+SELECT+1 HTTP/1.1" 200 2048',
    "Mar 13 02:03:00 web sudo: www-data : COMMAND=/bin/bash",
    "Mar 13 02:03:05 web sudo: www-data : COMMAND=crontab -e",
    "Mar 13 02:04:00 web sshd[2000]: Accepted publickey for root from 192.168.1.50 port 51300",
    '2026-03-13T02:04:10,4624,Information,192.168.1.50,An account was successfully logged on',
    '{"timestamp":"2026-03-13T02:05:00","src_ip":"192.168.1.50","dst_ip":"evil.c2.io","event_type":"network_conn","dport":4444,"bytes":2048}',
    '{"timestamp":"2026-03-13T02:05:30","src_ip":"192.168.1.50","dst_ip":"evil.c2.io","event_type":"network_conn","dport":443,"bytes":820000}',
    '{"timestamp":"2026-03-13T02:06:00","src_ip":"192.168.1.50","dst_ip":"dropbox.attacker.io","event_type":"network_conn","dport":443,"bytes":15000000}',
    "Mar 13 02:10:00 web sudo: analyst1 : COMMAND=/usr/bin/find / -name '*.pem'",
    '{"timestamp":"2026-03-13T02:10:30","src_ip":"10.0.0.200","dst_ip":"fileserver","event_type":"network_conn","dport":445,"bytes":25000000}',
]

class LogTailer:
    def __init__(self, filepath=None, simulate=True, speed=0.0):
        self.filepath=filepath; self.simulate=simulate or not filepath; self.speed=speed
        self._q=queue.Queue(); self._stop=threading.Event(); self._t=None
    def start(self):
        self._stop.clear()
        self._t=threading.Thread(target=self._simulate if self.simulate else self._tail,daemon=True)
        self._t.start()
    def stop(self): self._stop.set(); self._t and self._t.join(timeout=2)
    def get(self,timeout=0.1):
        try: return self._q.get(timeout=timeout)
        except queue.Empty: return None
    def _simulate(self):
        rng=random.Random(42)
        for line in SYNTHETIC_LOGS:
            if self._stop.is_set(): break
            self._q.put(line)
            if self.speed>0: time.sleep(self.speed*rng.uniform(0.7,1.3))
    def _tail(self):
        if not Path(self.filepath).exists(): return
        with open(self.filepath) as f:
            f.seek(0,2)
            while not self._stop.is_set():
                line=f.readline()
                if line: self._q.put(line)
                else: time.sleep(0.05)

# ── Pipeline ──────────────────────────────────────────────────

class IngestPipeline:
    def __init__(self,filepath=None,simulate=True,speed=0.0,window=8,ttl=300,score_fn=None):
        self.tailer=LogTailer(filepath,simulate,speed)
        self.parser=LogLineParser(); self.classifier=StageClassifier()
        self.tracker=SessionTracker(window,ttl); self.score_fn=score_fn; self._events=[]
    def start(self): self.tailer.start()
    def stop(self): self.tailer.stop()
    def sessions(self): return self.tracker.active_sessions()
    def stream(self,max_events=None,timeout_s=30):
        count=0; deadline=time.time()+timeout_s
        while True:
            if max_events and count>=max_events: break
            if time.time()>deadline: break
            raw=self.tailer.get(0.2)
            if raw is None: continue
            event=self.parser.parse(raw)
            if event is None: continue
            staged=self.classifier.classify(event)
            sequence=self.tracker.ingest(staged)
            sc,lv=(0,"SAFE")
            if self.score_fn and sequence: sc,lv=self.score_fn(sequence)
            self._events.append(staged); count+=1
            yield staged,sequence,sc,lv
    def save_events(self,path=None):
        out=path or ING_DIR/"ingested_events.json"
        data=[{"timestamp":e.event.timestamp,"src_ip":e.event.src_ip,
               "event_type":e.event.event_type,"stage":e.stage,
               "confidence":round(e.confidence,3),"rule":e.rule_matched}
              for e in self._events]
        with open(out,"w") as f: json.dump(data,f,indent=2)
        logger.info(f"Saved {len(data)} events -> {out}")
        return out

# ── Demo ──────────────────────────────────────────────────────

def _score(seq):
    BASE={"Reconnaissance":20,"Credential Access":40,"Exploitation":55,"Persistence":55,
          "Privilege Escalation":65,"Lateral Movement":70,"Command & Control":75,
          "Data Exfiltration":80,"Impact":80}
    KC=list(BASE.keys())
    s=seq[-1] if seq else "Reconnaissance"
    base=BASE.get(s,30); depth=max((KC.index(x)+1 for x in seq if x in KC),default=0)
    sc=min(100,base+(depth/len(KC))*10+5)
    lv="CRITICAL" if sc>=85 else "HIGH" if sc>=70 else "MEDIUM" if sc>=50 else "LOW" if sc>=30 else "SAFE"
    return sc,lv

if __name__=="__main__":
    import sys; sys.path.insert(0,str(BASE_DIR))
    print(f"\n{'='*62}\n  MODULE 9 — Real-time Log Ingestion\n{'='*62}\n")
    pipeline=IngestPipeline(simulate=True,speed=0.0,score_fn=_score)
    pipeline.start()
    ICON={"SAFE":"[SAFE]","LOW":"[LOW]","MEDIUM":"[MED]","HIGH":"[HIGH]","CRITICAL":"[CRIT]"}
    print(f"  {'Timestamp':<20} {'Src IP':<16} {'Stage':<25} {'Scr':>4} {'Level'}")
    print(f"  {'─'*76}")
    prev={}
    for staged,seq,sc,lv in pipeline.stream(max_events=18,timeout_s=10):
        ts=staged.event.timestamp[-8:] if len(staged.event.timestamp)>8 else staged.event.timestamp
        print(f"  {ts:<20} {staged.event.src_ip:<16} {staged.stage:<25} {sc:>4.0f}  {ICON[lv]}")
        print(f"    rule={staged.rule_matched:<30} seq=[{' -> '.join(seq)}]")
        curr=pipeline.sessions()
        for ip,s in curr.items():
            if len(s)>=3 and s!=prev.get(ip):
                _,l=_score(s)
                if l in ("HIGH","CRITICAL"):
                    print(f"\n  !! SESSION ALERT [{ip}]: {' -> '.join(s)} [{l}]\n")
        prev={ip:list(s) for ip,s in curr.items()}
    pipeline.stop()
    out=pipeline.save_events()
    print(f"\n  Active sessions:")
    for ip,s in pipeline.sessions().items():
        sc2,lv2=_score(s); print(f"    {ip:<16} {' -> '.join(s)}  [{lv2} {sc2:.0f}]")
    print(f"\n  Events -> {out}\n  Module 9 complete.\n{'='*62}\n")
