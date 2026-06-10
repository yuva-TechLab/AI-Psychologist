"""
Module 5: Threat Scoring Engine
================================
Converts AI model predictions into actionable risk scores and
recommended defensive responses.

Scoring formula
---------------
  base_score      = STAGE_BASE_SCORES[predicted_stage]      (0–80)
  confidence_boost= confidence * CONFIDENCE_WEIGHT          (0–10)
  sequence_boost  = progression_depth * DEPTH_WEIGHT        (0–10)
  final_score     = clip(base + confidence_boost + seq_boost, 0, 100)

Risk levels
-----------
  SAFE     :  0 – 29
  LOW      : 30 – 49
  MEDIUM   : 50 – 69
  HIGH     : 70 – 84
  CRITICAL : 85 – 100

Educational use only — defensive research prototype.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SEQ_DIR  = BASE_DIR / "data" / "sequences"
MDL_DIR  = BASE_DIR / "models"

# ─────────────────────────────────────────────
# SCORING TABLES
# ─────────────────────────────────────────────

# Base risk score per MITRE ATT&CK stage (0–80)
STAGE_BASE_SCORES = {
    "Benign"              :  0,
    "Reconnaissance"      : 20,
    "Discovery"           : 25,
    "Credential Access"   : 40,
    "Exploitation"        : 55,
    "Persistence"         : 55,
    "Privilege Escalation": 65,
    "Defense Evasion"     : 65,
    "Lateral Movement"    : 70,
    "Command & Control"   : 75,
    "Collection"          : 75,
    "Exfiltration"        : 80,
    "Data Exfiltration"   : 80,
    "Impact"              : 80,
    "Unknown"             : 30,
}

# Weight constants
CONFIDENCE_WEIGHT = 10.0   # max +10 points from model confidence
DEPTH_WEIGHT      = 10.0   # max +10 points from kill-chain depth

# Kill-chain depth order (later = deeper = higher risk)
KILL_CHAIN_DEPTH = [
    "Reconnaissance", "Discovery", "Credential Access",
    "Exploitation", "Persistence", "Privilege Escalation",
    "Defense Evasion", "Lateral Movement", "Command & Control",
    "Collection", "Data Exfiltration", "Exfiltration", "Impact",
]
MAX_DEPTH = len(KILL_CHAIN_DEPTH)

# Risk level thresholds
RISK_THRESHOLDS = [
    (85, "CRITICAL", "🔴"),
    (70, "HIGH",     "🟠"),
    (50, "MEDIUM",   "🟡"),
    (30, "LOW",      "🔵"),
    ( 0, "SAFE",     "🟢"),
]

# Recommended actions per predicted stage
STAGE_DEFENSES = {
    "Benign"              : "Continue baseline monitoring. No action needed.",
    "Reconnaissance"      : "Block port scanning sources. Enable IDS signatures for scan patterns. Rate-limit ICMP.",
    "Discovery"           : "Alert on AD enumeration commands (net user, whoami). Restrict LDAP queries.",
    "Credential Access"   : "Enable MFA immediately. Lock accounts after 5 failed attempts. Alert SOC on brute-force.",
    "Exploitation"        : "Patch affected services urgently. Deploy WAF rules. Activate honeypots.",
    "Persistence"         : "Audit startup items, scheduled tasks, and registry run keys. Enable FIM.",
    "Privilege Escalation": "Enforce least-privilege. Monitor sudo/admin activity. Deploy UEBA alerts.",
    "Defense Evasion"     : "Increase logging verbosity. Alert on log clearing events. Deploy EDR.",
    "Lateral Movement"    : "Segment network immediately. Monitor SMB/RDP/WMI. Disable unnecessary shares.",
    "Command & Control"   : "Block suspicious outbound connections. Monitor DNS beaconing. Isolate host.",
    "Collection"          : "Monitor file access patterns. Alert on bulk reads. Restrict USB/cloud sync.",
    "Data Exfiltration"   : "Activate DLP rules. Block large outbound transfers. Isolate affected systems NOW.",
    "Exfiltration"        : "Activate DLP rules. Block large outbound transfers. Isolate affected systems NOW.",
    "Impact"              : "ISOLATE affected systems immediately. Activate IR plan. Restore from clean backup.",
    "Unknown"             : "Escalate to SOC for manual investigation. Increase log collection.",
}

# MITRE ATT&CK technique examples per stage
STAGE_TECHNIQUES = {
    "Reconnaissance"      : "T1595 (Active Scanning), T1592 (Gather Victim Info)",
    "Credential Access"   : "T1110 (Brute Force), T1555 (Credentials from Password Stores)",
    "Exploitation"        : "T1190 (Exploit Public-Facing App), T1203 (Exploitation for Client Exec)",
    "Privilege Escalation": "T1548 (Abuse Elevation Control), T1134 (Access Token Manipulation)",
    "Lateral Movement"    : "T1021 (Remote Services), T1570 (Lateral Tool Transfer)",
    "Command & Control"   : "T1071 (App Layer Protocol), T1095 (Non-App Layer Protocol)",
    "Data Exfiltration"   : "T1041 (Exfil Over C2 Channel), T1048 (Exfil Over Alt Protocol)",
    "Impact"              : "T1486 (Data Encrypted for Impact), T1485 (Data Destruction)",
    "Persistence"         : "T1053 (Scheduled Task/Job), T1547 (Boot/Logon Autostart)",
    "Discovery"           : "T1083 (File & Dir Discovery), T1087 (Account Discovery)",
}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ThreatScore:
    """Full scoring result returned by the engine."""
    # Input context
    input_sequence    : list
    detected_stage    : str

    # Prediction
    predicted_stage   : str
    confidence        : float
    top_predictions   : list   # [(stage, prob), ...]

    # Scoring
    raw_score         : float
    base_score        : int
    confidence_boost  : float
    sequence_boost    : float
    risk_level        : str
    risk_emoji        : str

    # Response
    recommended_action: str
    mitre_techniques  : str
    alert_message     : str

    def to_dict(self) -> dict:
        return {
            "input_sequence"    : self.input_sequence,
            "detected_stage"    : self.detected_stage,
            "predicted_stage"   : self.predicted_stage,
            "confidence"        : round(self.confidence, 3),
            "raw_score"         : round(self.raw_score, 1),
            "risk_level"        : self.risk_level,
            "recommended_action": self.recommended_action,
            "alert_message"     : self.alert_message,
            "top_predictions"   : self.top_predictions,
        }

    def __str__(self) -> str:
        bar = "█" * int(self.raw_score / 5)
        return (
            f"\n{'─'*56}\n"
            f"  THREAT ASSESSMENT\n"
            f"{'─'*56}\n"
            f"  Detected stage    : {self.detected_stage}\n"
            f"  Predicted next    : {self.predicted_stage}\n"
            f"  Confidence        : {self.confidence:.1%}\n"
            f"  Threat score      : {self.raw_score:.0f}/100  {self.risk_emoji} {self.risk_level}\n"
            f"  Score bar         : [{bar:<20}]\n"
            f"  MITRE techniques  : {self.mitre_techniques}\n"
            f"\n  ⚡ RECOMMENDED ACTION:\n"
            f"  {self.recommended_action}\n"
            f"\n  ⚠  ALERT:\n"
            f"  {self.alert_message}\n"
            f"{'─'*56}\n"
        )


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

class ThreatScoringEngine:
    """
    Converts AI model output → structured threat scores and alerts.

    Usage
    -----
    engine = ThreatScoringEngine()
    engine.load_model("transformer")   # or "lstm" or "markov"

    result = engine.score(
        sequence=["Reconnaissance", "Credential Access", "Exploitation"]
    )
    print(result)

    # Batch scoring
    results = engine.score_batch(session_list)
    """

    def __init__(self):
        self.model     = None
        self.vocab     = {}
        self.inv_vocab = {}
        self.model_name = "none"

    # ── Load model ───────────────────────────

    def load_model(self, model_type: str = "transformer") -> None:
        """Load a trained model from models/. model_type: 'transformer'|'lstm'|'markov'"""
        import sys
        sys.path.insert(0, str(BASE_DIR))

        vocab_path = SEQ_DIR / "vocabulary.json"
        if not vocab_path.exists():
            raise FileNotFoundError("Run Module 3 first to generate vocabulary.json")

        with open(vocab_path) as f:
            vdata = json.load(f)
        self.vocab     = {k: int(v) for k, v in vdata["vocab"].items()}
        self.inv_vocab = {int(k): v for k, v in vdata["inv_vocab"].items()}

        if model_type == "markov":
            from training.train_models import MarkovChainModel
            sessions_path = SEQ_DIR / "attack_sessions.json"
            with open(sessions_path) as f:
                sessions = json.load(f)
            self.model = MarkovChainModel(order=1)
            self.model.fit(sessions, self.vocab)

        elif model_type in ("lstm", "transformer"):
            from training.train_models import LSTMPredictor, TransformerPredictor
            meta_path = MDL_DIR / f"{model_type}_attack_predictor_meta.json"
            npz_path  = MDL_DIR / f"{model_type}_attack_predictor.npz"

            if not npz_path.exists():
                raise FileNotFoundError(f"Run Module 4 first. Model not found: {npz_path}")

            with open(meta_path) as f:
                meta = json.load(f)

            vs = meta.get("vocab_size", len(self.vocab))

            if model_type == "lstm":
                m = LSTMPredictor(vocab_size=vs, embed_dim=32,
                                  hidden_dim=64, num_classes=vs)
                w = np.load(npz_path)
                m.embed = w["embed"]; m.W1=w["W1"]; m.b1=w["b1"]
                m.W2=w["W2"]; m.b2=w["b2"]
            else:
                m = TransformerPredictor(vocab_size=vs, embed_dim=32,
                                         nhead=4, num_layers=2, dim_ff=64,
                                         num_classes=vs)
                w = np.load(npz_path)
                m.embed=w["embed"]; m.pos=w["pos"]
                m.W1=w["W1"]; m.b1=w["b1"]
                m.W2=w["W2"]; m.b2=w["b2"]
                for bi, block in enumerate(m.blocks):
                    block.W1=w[f"block{bi}_W1"]; block.b1=w[f"block{bi}_b1"]
                    block.W2=w[f"block{bi}_W2"]; block.b2=w[f"block{bi}_b2"]

            self.model = m

        self.model_name = model_type
        logger.info(f"Loaded model: {model_type}")

    # ── Core scoring ─────────────────────────

    def score(
        self,
        sequence : list,
        window   : int = 3,
    ) -> ThreatScore:
        """
        Score a stage sequence and return a full ThreatScore.

        sequence: list of stage name strings
                  e.g. ["Reconnaissance", "Credential Access", "Exploitation"]
        """
        if not sequence:
            raise ValueError("sequence must contain at least one stage")

        detected_stage = sequence[-1]

        # Encode to tokens
        tokens = [self.vocab.get(s, self.vocab.get("<UNK>", 1))
                  for s in sequence[-window:]]

        # Model prediction
        pred_token, confidence, raw_topk = self.model.predict(tokens, top_k=5)
        predicted_stage = self.inv_vocab.get(pred_token, "Unknown")

        # Top-k readable
        top_predictions = [
            (self.inv_vocab.get(tok, "?"), round(float(prob), 3))
            for tok, prob in raw_topk
            if self.inv_vocab.get(tok, "?") not in ("<PAD>", "<UNK>")
        ]

        # ── Scoring formula ───────────────────
        base_score = STAGE_BASE_SCORES.get(predicted_stage,
                     STAGE_BASE_SCORES.get(detected_stage, 30))

        confidence_boost = confidence * CONFIDENCE_WEIGHT

        # Sequence depth boost: how far along the kill chain?
        depth = 0
        for stage in sequence:
            try:
                d = KILL_CHAIN_DEPTH.index(stage) + 1
                depth = max(depth, d)
            except ValueError:
                pass
        sequence_boost = (depth / MAX_DEPTH) * DEPTH_WEIGHT

        raw_score = np.clip(base_score + confidence_boost + sequence_boost, 0, 100)

        # ── Risk level ────────────────────────
        risk_level = "SAFE"
        risk_emoji = "🟢"
        for threshold, level, emoji in RISK_THRESHOLDS:
            if raw_score >= threshold:
                risk_level = level
                risk_emoji = emoji
                break

        # ── Recommended action ────────────────
        action = STAGE_DEFENSES.get(
            predicted_stage,
            STAGE_DEFENSES.get(detected_stage, "Escalate to SOC for investigation.")
        )

        techniques = STAGE_TECHNIQUES.get(predicted_stage, "See MITRE ATT&CK for details")

        # ── Alert message ─────────────────────
        alert = self._build_alert(
            sequence, detected_stage, predicted_stage,
            confidence, raw_score, risk_level
        )

        return ThreatScore(
            input_sequence    = sequence,
            detected_stage    = detected_stage,
            predicted_stage   = predicted_stage,
            confidence        = confidence,
            top_predictions   = top_predictions,
            raw_score         = float(raw_score),
            base_score        = base_score,
            confidence_boost  = round(confidence_boost, 2),
            sequence_boost    = round(sequence_boost, 2),
            risk_level        = risk_level,
            risk_emoji        = risk_emoji,
            recommended_action= action,
            mitre_techniques  = techniques,
            alert_message     = alert,
        )

    def _build_alert(
        self, sequence, detected, predicted, confidence, score, level
    ) -> str:
        seq_str = " → ".join(sequence)
        return (
            f"[{level}] Attack progression detected: {seq_str}. "
            f"AI predicts next step is '{predicted}' "
            f"(confidence {confidence:.0%}, score {score:.0f}/100). "
            f"Immediate defensive action required."
        )

    # ── Batch scoring ─────────────────────────

    def score_batch(self, sessions: list) -> list:
        """Score a list of sessions, return list of ThreatScore objects."""
        results = []
        for session in sessions:
            if len(session) < 2:
                continue
            # Score at each step in the session (simulate real-time)
            for end in range(2, len(session) + 1):
                result = self.score(session[:end])
                results.append(result)
        logger.info(f"Batch scored {len(results)} events from {len(sessions)} sessions")
        return results

    # ── Distribution stats ────────────────────

    def score_distribution(self, results: list) -> dict:
        scores = [r.raw_score for r in results]
        levels = [r.risk_level for r in results]
        from collections import Counter
        level_counts = Counter(levels)

        return {
            "mean"   : round(float(np.mean(scores)), 2),
            "median" : round(float(np.median(scores)), 2),
            "max"    : round(float(np.max(scores)), 2),
            "min"    : round(float(np.min(scores)), 2),
            "levels" : dict(level_counts),
            "scores" : scores,
        }


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    engine = ThreatScoringEngine()
    engine.load_model("transformer")

    print(f"\n{'='*56}")
    print(f"  MODULE 5 — Threat Scoring Engine")
    print(f"{'='*56}\n")

    # ── Test scenarios ────────────────────────
    scenarios = [
        {
            "name"    : "Early-stage recon",
            "sequence": ["Reconnaissance"],
        },
        {
            "name"    : "Credential attack in progress",
            "sequence": ["Reconnaissance", "Credential Access"],
        },
        {
            "name"    : "Post-exploit privilege escalation",
            "sequence": ["Reconnaissance", "Credential Access", "Exploitation"],
        },
        {
            "name"    : "Advanced — lateral movement detected",
            "sequence": ["Reconnaissance", "Credential Access", "Exploitation",
                         "Privilege Escalation", "Lateral Movement"],
        },
        {
            "name"    : "Critical — data exfiltration imminent",
            "sequence": ["Reconnaissance", "Exploitation", "Privilege Escalation",
                         "Command & Control"],
        },
    ]

    all_results = []
    for sc in scenarios:
        print(f"  Scenario: {sc['name']}")
        result = engine.score(sc["sequence"])
        print(result)
        all_results.append(result)

    # ── Distribution summary ──────────────────
    dist = engine.score_distribution(all_results)
    print(f"\n  Score distribution across {len(all_results)} scenarios:")
    print(f"  Mean={dist['mean']}  Median={dist['median']}")
    print(f"  Min={dist['min']}    Max={dist['max']}")
    print(f"\n  Risk level breakdown:")
    for level, count in sorted(dist["levels"].items()):
        print(f"    {level:<12}: {count}")

    print(f"\nModule 5 complete.")
    print(f"Next: Module 6 — Streamlit Security Dashboard\n")
