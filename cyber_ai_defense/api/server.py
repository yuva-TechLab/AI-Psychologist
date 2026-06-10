"""
Module 12: REST API
=====================
Exposes the full CyberAI Defense pipeline as a REST API.

Two implementations in one file:
  - FastAPIServer  : production-grade (requires: pip install fastapi uvicorn)
  - BuiltinServer  : zero-dependency fallback using http.server (runs anywhere)

Both expose identical endpoints and JSON schemas.

Endpoints
---------
  POST /score          — score a sequence, return threat level + prediction
  POST /enrich         — enrich an IP with GeoIP + reputation + score adjust
  POST /explain        — XAI explanation for a sequence + prediction
  POST /ingest         — parse a raw log line → stage + sequence + score
  POST /alert          — dispatch an alert to configured channels
  GET  /sessions       — list all active sessions from the tracker
  GET  /health         — liveness check
  GET  /docs           — interactive API docs (FastAPI only)

Run (FastAPI / production):
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Run (built-in / zero-dep):
  python api/server.py

Educational use only — defensive research prototype.
"""

import json, sys, time, traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from prediction.threat_scorer   import ThreatScoringEngine
from xai.explainer               import XAIExplainer
from data.log_ingestor import IngestPipeline, LogLineParser, StageClassifier, SessionTracker
from intel.threat_intel          import IPEnricher
from alerts.alert_manager        import AlertManager, AlertConfig

# ─────────────────────────────────────────────
# SHARED PIPELINE SERVICES  (singleton pattern)
# ─────────────────────────────────────────────

class Services:
    """Lazy-initialised singletons shared across all request handlers."""
    _scorer    = None
    _explainer = None
    _parser    = None
    _classifier= None
    _tracker   = None
    _enricher  = None
    _alertmgr  = None

    @classmethod
    def scorer(cls):
        if not cls._scorer:
            cls._scorer = ThreatScoringEngine()
            cls._scorer.load_model("markov")
        return cls._scorer

    @classmethod
    def explainer(cls):
        if not cls._explainer:
            cls._explainer = XAIExplainer()
        return cls._explainer

    @classmethod
    def parser(cls):
        if not cls._parser:
            cls._parser = LogLineParser()
        return cls._parser

    @classmethod
    def classifier(cls):
        if not cls._classifier:
            cls._classifier = StageClassifier()
        return cls._classifier

    @classmethod
    def tracker(cls):
        if not cls._tracker:
            cls._tracker = SessionTracker(window=10, ttl=300)
        return cls._tracker

    @classmethod
    def enricher(cls):
        if not cls._enricher:
            cls._enricher = IPEnricher(dry_run=True)
        return cls._enricher

    @classmethod
    def alertmgr(cls):
        if not cls._alertmgr:
            cfg = AlertConfig(dry_run=True, cooldown_s=0,
                              enabled_channels=["slack","email","siem"])
            cls._alertmgr = AlertManager(cfg)
        return cls._alertmgr


# ─────────────────────────────────────────────
# CORE HANDLER LOGIC  (shared by both servers)
# ─────────────────────────────────────────────

KILL_CHAIN = [
    "Reconnaissance","Discovery","Credential Access","Exploitation",
    "Persistence","Privilege Escalation","Defense Evasion","Lateral Movement",
    "Command & Control","Collection","Data Exfiltration","Impact",
]

def _score_sequence(sequence: list) -> dict:
    """POST /score — score a sequence and predict next stage."""
    if not sequence or not isinstance(sequence, list):
        return {"error": "sequence must be a non-empty list of stage strings"}, 400

    scorer  = Services.scorer()
    threat  = scorer.score(sequence)
    return {
        "sequence"       : sequence,
        "detected_stage" : threat.detected_stage,
        "predicted_stage": threat.predicted_stage,
        "confidence"     : round(threat.confidence, 3),
        "top_predictions": [[s, round(p,3)] for s,p in threat.top_predictions[:5]],
        "threat_score"   : round(threat.raw_score, 1),
        "risk_level"     : threat.risk_level,
        "risk_emoji"     : threat.risk_emoji,
        "recommended_action": threat.recommended_action,
        "mitre_techniques": threat.mitre_techniques,
        "alert_message"  : threat.alert_message,
        "score_breakdown": {
            "base_score"      : threat.base_score,
            "confidence_boost": round(threat.confidence_boost, 2),
            "sequence_boost"  : round(threat.sequence_boost, 2),
        },
    }, 200


def _enrich_ip(ip: str, base_score: float = 50.0, sequence: list = None) -> dict:
    """POST /enrich — GeoIP + reputation + score adjustment for an IP."""
    if not ip:
        return {"error": "ip is required"}, 400
    enricher = Services.enricher()
    result   = enricher.enrich_session(ip, base_score, sequence or [])
    return result, 200


def _explain_sequence(sequence: list, predicted: str, confidence: float) -> dict:
    """POST /explain — XAI explanation for a prediction."""
    if not sequence or not predicted:
        return {"error": "sequence and predicted are required"}, 400
    explainer = Services.explainer()
    exp       = explainer.explain(sequence, predicted, confidence)
    return exp.to_dict(), 200


def _ingest_line(raw_line: str, src_ip: str = None) -> dict:
    """POST /ingest — parse a raw log line, classify stage, score session."""
    if not raw_line:
        return {"error": "raw_line is required"}, 400
    parser     = Services.parser()
    classifier = Services.classifier()
    tracker    = Services.tracker()
    scorer     = Services.scorer()
    enricher   = Services.enricher()

    event  = parser.parse(raw_line, default_src=src_ip or "0.0.0.0")
    if not event:
        return {"error": "Could not parse log line"}, 422

    staged   = classifier.classify(event)
    sequence = tracker.ingest(staged)

    threat   = scorer.score(sequence)
    adj_score, adj, flags = enricher.adjust_score(threat.raw_score, event.src_ip)
    adj_lv   = ("CRITICAL" if adj_score>=85 else "HIGH" if adj_score>=70
                else "MEDIUM" if adj_score>=50 else "LOW" if adj_score>=30 else "SAFE")

    return {
        "src_ip"         : event.src_ip,
        "event_type"     : event.event_type,
        "log_format"     : event.log_format,
        "stage"          : staged.stage,
        "rule_matched"   : staged.rule_matched,
        "confidence"     : round(staged.confidence, 3),
        "sequence"       : sequence,
        "base_score"     : round(threat.raw_score, 1),
        "adjusted_score" : round(adj_score, 1),
        "risk_level"     : adj_lv,
        "intel_flags"    : flags,
        "intel_adjustments": adj,
        "predicted_next" : threat.predicted_stage,
        "recommended_action": threat.recommended_action,
    }, 200


def _dispatch_alert(body: dict) -> dict:
    """POST /alert — dispatch an alert through configured channels."""
    required = ["src_ip","sequence","detected","predicted","confidence","score","level"]
    missing  = [k for k in required if k not in body]
    if missing:
        return {"error": f"Missing fields: {missing}"}, 400
    mgr    = Services.alertmgr()
    result = mgr.dispatch(
        src_ip     = body["src_ip"],
        sequence   = body["sequence"],
        detected   = body["detected"],
        predicted  = body["predicted"],
        confidence = float(body["confidence"]),
        score      = float(body["score"]),
        level      = body["level"],
        rule       = body.get("rule", "api"),
    )
    return result, 200


def _get_sessions() -> dict:
    """GET /sessions — list all active tracked sessions."""
    tracker  = Services.tracker()
    sessions = tracker.active_sessions()
    scorer   = Services.scorer()
    result   = {}
    for ip, seq in sessions.items():
        threat = scorer.score(seq)
        result[ip] = {
            "sequence"   : seq,
            "threat_score": round(threat.raw_score, 1),
            "risk_level" : threat.risk_level,
            "predicted"  : threat.predicted_stage,
        }
    return {"sessions": result, "count": len(result)}, 200


def _health() -> dict:
    """GET /health"""
    return {
        "status" : "ok",
        "version": "1.0.0",
        "modules": ["scorer","explainer","ingestor","enricher","alertmgr"],
        "uptime" : round(time.time() - _START_TIME, 1),
    }, 200


_START_TIME = time.time()

ROUTES = {
    ("POST", "/score")  : lambda b, _: _score_sequence(b.get("sequence", [])),
    ("POST", "/enrich") : lambda b, _: _enrich_ip(b.get("ip",""), b.get("base_score",50.0), b.get("sequence")),
    ("POST", "/explain"): lambda b, _: _explain_sequence(b.get("sequence",[]), b.get("predicted",""), float(b.get("confidence",0.8))),
    ("POST", "/ingest") : lambda b, _: _ingest_line(b.get("raw_line",""), b.get("src_ip")),
    ("POST", "/alert")  : lambda b, _: _dispatch_alert(b),
    ("GET",  "/sessions"): lambda _, __: _get_sessions(),
    ("GET",  "/health") : lambda _, __: _health(),
}


# ─────────────────────────────────────────────
# BUILT-IN HTTP SERVER  (zero dependencies)
# ─────────────────────────────────────────────

class APIHandler(BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 JSON handler."""

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def _send(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _dispatch(self, method: str):
        path    = urlparse(self.path).path.rstrip("/") or "/"
        handler = ROUTES.get((method, path))
        if not handler:
            self._send({"error": f"No route for {method} {path}",
                        "available": [f"{m} {p}" for m,p in ROUTES]}, 404)
            return
        try:
            body   = self._read_body() if method == "POST" else {}
            result, status = handler(body, self.headers)
            self._send(result, status)
        except Exception as e:
            self._send({"error": str(e), "trace": traceback.format_exc()[-500:]}, 500)

    def do_GET(self):  self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class BuiltinServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port

    def run(self):
        server = HTTPServer((self.host, self.port), APIHandler)
        logger_print(f"CyberAI Defense API running on http://{self.host}:{self.port}")
        logger_print(f"Endpoints: {[f'{m} {p}' for m,p in ROUTES.keys()]}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger_print("Server stopped.")


def logger_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────────
# FASTAPI SERVER  (production — needs: pip install fastapi uvicorn)
# ─────────────────────────────────────────────

FASTAPI_APP_CODE = '''
"""
FastAPI production server — drop-in replacement for BuiltinServer.
Run: uvicorn api.server_fastapi:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.server import (Services, _score_sequence, _enrich_ip, _explain_sequence,
                         _ingest_line, _dispatch_alert, _get_sessions, _health)

app = FastAPI(title="CyberAI Defense API", version="1.0.0",
              description="Generative AI Cyber Crime Psychologist — REST API")

class ScoreReq(BaseModel):
    sequence: list[str]

class EnrichReq(BaseModel):
    ip: str; base_score: float = 50.0; sequence: list[str] = []

class ExplainReq(BaseModel):
    sequence: list[str]; predicted: str; confidence: float = 0.8

class IngestReq(BaseModel):
    raw_line: str; src_ip: Optional[str] = None

class AlertReq(BaseModel):
    src_ip: str; sequence: list[str]; detected: str; predicted: str
    confidence: float; score: float; level: str; rule: str = "api"

@app.post("/score")   
def score(req: ScoreReq):   
    r,s = _score_sequence(req.sequence);   return r if s==200 else HTTPException(s,r)

@app.post("/enrich")  
def enrich(req: EnrichReq):  
    r,s = _enrich_ip(req.ip, req.base_score, req.sequence); return r if s==200 else HTTPException(s,r)

@app.post("/explain") 
def explain(req: ExplainReq): 
    r,s = _explain_sequence(req.sequence, req.predicted, req.confidence); return r if s==200 else HTTPException(s,r)

@app.post("/ingest")  
def ingest(req: IngestReq):  
    r,s = _ingest_line(req.raw_line, req.src_ip); return r if s==200 else HTTPException(s,r)

@app.post("/alert")   
def alert(req: AlertReq):   
    r,s = _dispatch_alert(req.dict()); return r if s==200 else HTTPException(s,r)

@app.get("/sessions") 
def sessions():              
    r,s = _get_sessions(); return r

@app.get("/health")   
def health():                
    r,s = _health(); return r
'''


# ─────────────────────────────────────────────
# DEMO  (test all endpoints without a real server)
# ─────────────────────────────────────────────

def run_demo():
    print(f"\n{'='*60}")
    print(f"  MODULE 12 — REST API  (endpoint demo)")
    print(f"{'='*60}\n")

    tests = [
        ("GET",  "/health",  {}),
        ("POST", "/score",   {"sequence": ["Reconnaissance","Credential Access","Exploitation"]}),
        ("POST", "/enrich",  {"ip":"192.168.1.50","base_score":78.0,"sequence":["Recon","Exploit"]}),
        ("POST", "/explain", {"sequence":["Reconnaissance","Credential Access","Exploitation"],
                              "predicted":"Privilege Escalation","confidence":0.99}),
        ("POST", "/ingest",  {"raw_line":"Mar 13 02:01:00 sshd[1234]: Failed password for root from 10.10.0.5 port 51234",
                              "src_ip":"10.10.0.5"}),
        ("POST", "/alert",   {"src_ip":"172.16.0.55","sequence":["Lateral Movement","Command & Control"],
                              "detected":"Command & Control","predicted":"Data Exfiltration",
                              "confidence":0.96,"score":92.0,"level":"CRITICAL","rule":"https_beacon"}),
        ("GET",  "/sessions",{}),
    ]

    handler_map = {
        ("GET",  "/health")  : lambda b: _health(),
        ("POST", "/score")   : lambda b: _score_sequence(b.get("sequence",[])),
        ("POST", "/enrich")  : lambda b: _enrich_ip(b.get("ip",""),b.get("base_score",50),b.get("sequence")),
        ("POST", "/explain") : lambda b: _explain_sequence(b.get("sequence",[]),b.get("predicted",""),float(b.get("confidence",0.8))),
        ("POST", "/ingest")  : lambda b: _ingest_line(b.get("raw_line",""),b.get("src_ip")),
        ("POST", "/alert")   : lambda b: _dispatch_alert(b),
        ("GET",  "/sessions"): lambda b: _get_sessions(),
    }

    for method, path, body in tests:
        print(f"  {method} {path}")
        t0 = time.time()
        result, status = handler_map[(method, path)](body)
        ms = (time.time()-t0)*1000
        ok = "✓" if status == 200 else "✗"
        print(f"  {ok} HTTP {status}  ({ms:.1f}ms)")

        # Print a meaningful excerpt of each response
        if path == "/health":
            print(f"    status={result['status']}  uptime={result['uptime']}s")
        elif path == "/score":
            print(f"    detected={result.get('detected_stage')}  "
                  f"predicted={result.get('predicted_stage')}  "
                  f"score={result.get('threat_score')}  "
                  f"level={result.get('risk_level')}")
        elif path == "/enrich":
            print(f"    country={result.get('country')}  "
                  f"abuse={result.get('geo',{}).get('abuseipdb_score')}  "
                  f"base={result.get('base_score')}→adj={result.get('adjusted_score')}  "
                  f"delta={result.get('score_delta'):+.0f}")
        elif path == "/explain":
            print(f"    prediction={result.get('predicted_stage')}  "
                  f"score={result.get('threat_score')}  "
                  f"top_contrib={max(result.get('context_weights',{}).items(), key=lambda x:x[1], default=('—',0))[0]}")
        elif path == "/ingest":
            print(f"    src={result.get('src_ip')}  "
                  f"stage={result.get('stage')}  "
                  f"rule={result.get('rule_matched')}  "
                  f"score={result.get('adjusted_score')}")
        elif path == "/alert":
            print(f"    sent={result.get('sent')}  "
                  f"channels={result.get('channels')}")
        elif path == "/sessions":
            count = result.get('count', 0)
            print(f"    active_sessions={count}")
            for ip, s in result.get("sessions",{}).items():
                print(f"      {ip}: {s['sequence']} [{s['risk_level']} {s['threat_score']}]")
        print()

    # Save FastAPI app
    fastapi_path = BASE_DIR / "api" / "server_fastapi.py"
    fastapi_path.write_text(FASTAPI_APP_CODE)

    print(f"  {'─'*56}")
    print(f"  ENDPOINT SUMMARY")
    print(f"  {'─'*56}")
    routes = [
        ("POST", "/score",    "Score a stage sequence → threat level + prediction"),
        ("POST", "/enrich",   "GeoIP + reputation + score adjustment for an IP"),
        ("POST", "/explain",  "XAI explanation for a sequence + prediction"),
        ("POST", "/ingest",   "Parse raw log line → stage + sequence + score"),
        ("POST", "/alert",    "Dispatch alert to Slack / email / SIEM"),
        ("GET",  "/sessions", "List all active tracked sessions"),
        ("GET",  "/health",   "Liveness check"),
    ]
    for method, path, desc in routes:
        print(f"  {method:<5} {path:<12} {desc}")
    print(f"\n  Run (built-in, zero-dep):  python api/server.py")
    print(f"  Run (FastAPI/production):  uvicorn api.server:app --port 8000")
    print(f"\n  FastAPI app saved → {fastapi_path}")
    print(f"\n  Module 12 complete.\n{'='*60}\n")


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)   # quiet service init logs in demo

    if "--serve" in sys.argv:
        BuiltinServer().run()
    else:
        run_demo()
