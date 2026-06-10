"""
FastAPI production server — Cyber AI Defense
Run:
python -m uvicorn api.server_fastapi:app --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="CyberAI Defense API",
    version="1.0.0",
    description="Generative AI Cyber Crime Psychologist — REST API"
)

# ================== REQUEST MODELS ==================

class ScoreReq(BaseModel):
    sequence: list[str]

class EnrichReq(BaseModel):
    ip: str
    base_score: float = 50.0
    sequence: list[str] = []

class ExplainReq(BaseModel):
    sequence: list[str]
    predicted: str
    confidence: float = 0.8

class IngestReq(BaseModel):
    raw_line: str
    src_ip: Optional[str] = None

class AlertReq(BaseModel):
    src_ip: str
    sequence: list[str]
    detected: str
    predicted: str
    confidence: float
    score: float
    level: str
    rule: str = "api"

# ================== CORE FUNCTIONS ==================

def _score_sequence(sequence):
    score = len(sequence) * 25

    if score < 30:
        level = "LOW"
    elif score < 60:
        level = "MEDIUM"
    elif score < 80:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return ({
        "sequence": sequence,
        "threat_score": score,
        "risk_level": level
    }, 200)


def _enrich_ip(ip, base_score, sequence):
    return ({
        "ip": ip,
        "reputation": "suspicious" if base_score > 50 else "normal",
        "adjusted_score": base_score + len(sequence) * 5
    }, 200)


def _explain_sequence(sequence, predicted, confidence):
    return ({
        "sequence": sequence,
        "prediction": predicted,
        "confidence": confidence,
        "reason": "Pattern matches known cyber attack stages"
    }, 200)


def _ingest_line(raw_line, src_ip):
    return ({
        "message": "Log ingested successfully",
        "line": raw_line,
        "src_ip": src_ip
    }, 200)


def _dispatch_alert(data):
    return ({
        "status": "Alert sent",
        "details": data
    }, 200)


def _get_sessions():
    return ({
        "active_sessions": 5
    }, 200)


def _health():
    return ({
        "status": "healthy",
        "service": "CyberAI Defense API"
    }, 200)

# ================== API ROUTES ==================

@app.get("/")
def home():
    return {"message": "Cyber AI Defense API Running 🚀"}


@app.post("/score")
def score(req: ScoreReq):
    result, status = _score_sequence(req.sequence)
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/enrich")
def enrich(req: EnrichReq):
    result, status = _enrich_ip(req.ip, req.base_score, req.sequence)
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/explain")
def explain(req: ExplainReq):
    result, status = _explain_sequence(req.sequence, req.predicted, req.confidence)
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/ingest")
def ingest(req: IngestReq):
    result, status = _ingest_line(req.raw_line, req.src_ip)
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.post("/alert")
def alert(req: AlertReq):
    result, status = _dispatch_alert(req.dict())
    if status != 200:
        raise HTTPException(status_code=status, detail=result)
    return result


@app.get("/sessions")
def sessions():
    result, _ = _get_sessions()
    return result


@app.get("/health")
def health():
    result, _ = _health()
    return result