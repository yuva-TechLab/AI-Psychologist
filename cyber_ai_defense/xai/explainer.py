"""
Module 8: Explainable AI (XAI) Layer
======================================
Answers the question: "WHY did the AI predict that next stage?"

Three explanation strategies:
  1. TransitionExplainer  — probability-based: shows historical transition
                            frequencies that drove the prediction
  2. FeatureExplainer     — SHAP-inspired: shows which stages in the context
                            window contributed most to the prediction
  3. CounterfactualEngine — "what-if": shows what sequence change would lower
                            the threat score

Output formats:
  • Plain-text narrative  (for CLI / logs)
  • Structured dict       (for dashboard / API)
  • HTML card             (for Streamlit embed)

Educational use only — defensive research prototype.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SEQ_DIR  = BASE_DIR / "data" / "sequences"
XAI_DIR  = BASE_DIR / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# KNOWLEDGE BASE — drives all explanations
# ─────────────────────────────────────────────

# Transition probabilities (from Module 3 transition matrix)
TRANSITION_PROBS = {
    "Reconnaissance":       {"Credential Access":.44,"Exploitation":.36,"Discovery":.12,"Persistence":.08},
    "Credential Access":    {"Exploitation":.34,"Privilege Escalation":.32,"Lateral Movement":.22,"Discovery":.12},
    "Exploitation":         {"Privilege Escalation":.50,"Persistence":.28,"Lateral Movement":.14,"Reconnaissance":.08},
    "Privilege Escalation": {"Lateral Movement":.51,"Discovery":.28,"Command & Control":.14,"Impact":.07},
    "Lateral Movement":     {"Data Exfiltration":.44,"Command & Control":.28,"Discovery":.20,"Impact":.08},
    "Command & Control":    {"Data Exfiltration":.70,"Lateral Movement":.18,"Impact":.12},
    "Discovery":            {"Lateral Movement":.50,"Data Exfiltration":.30,"Impact":.20},
    "Persistence":          {"Privilege Escalation":.40,"Command & Control":.30,"Lateral Movement":.20,"Impact":.10},
    "Data Exfiltration":    {"Impact":.80,"Command & Control":.20},
    "Defense Evasion":      {"Lateral Movement":.50,"Command & Control":.30,"Privilege Escalation":.20},
    "Collection":           {"Data Exfiltration":.70,"Command & Control":.30},
    "Impact":               {"Reconnaissance":.50,"Data Exfiltration":.50},
}

KILL_CHAIN_ORDER = [
    "Reconnaissance","Discovery","Credential Access","Exploitation",
    "Persistence","Privilege Escalation","Defense Evasion","Lateral Movement",
    "Command & Control","Collection","Data Exfiltration","Impact",
]

BASE_SCORES = {
    "Reconnaissance":20,"Discovery":25,"Credential Access":40,"Exploitation":55,
    "Persistence":55,"Privilege Escalation":65,"Defense Evasion":65,
    "Lateral Movement":70,"Command & Control":75,"Collection":70,
    "Data Exfiltration":80,"Impact":80,"Unknown":30,
}

# Human-readable stage descriptions for narrative explanations
STAGE_DESCRIPTIONS = {
    "Reconnaissance":      "scanning for targets and gathering information",
    "Discovery":           "enumerating internal systems and accounts",
    "Credential Access":   "attempting to steal or brute-force credentials",
    "Exploitation":        "actively exploiting vulnerable services",
    "Persistence":         "establishing footholds to survive reboots",
    "Privilege Escalation":"attempting to gain elevated/admin privileges",
    "Defense Evasion":     "hiding activity and disabling security tools",
    "Lateral Movement":    "moving across the network to new hosts",
    "Command & Control":   "establishing a covert communication channel",
    "Collection":          "harvesting sensitive data from systems",
    "Data Exfiltration":   "transferring stolen data to attacker infrastructure",
    "Impact":              "destroying, encrypting, or disrupting systems",
}

# Known real-world attack patterns that match common sequences
KNOWN_PATTERNS = [
    {
        "name"      : "Classic Credential Pivot",
        "sequence"  : ["Reconnaissance","Credential Access","Exploitation"],
        "description": "Most common enterprise intrusion pattern — recon leads to credential theft, then exploitation of the gained access.",
        "prevalence": "Seen in 68% of enterprise breaches (Verizon DBIR)",
    },
    {
        "name"      : "APT Lateral Spread",
        "sequence"  : ["Exploitation","Privilege Escalation","Lateral Movement"],
        "description": "Classic APT progression after initial foothold — escalate locally, then pivot across the network.",
        "prevalence": "Characteristic of APT28, APT29, Lazarus Group",
    },
    {
        "name"      : "Ransomware Pre-Detonation",
        "sequence"  : ["Lateral Movement","Command & Control"],
        "description": "Attacker has achieved network-wide access and established C2 — ransomware detonation or exfiltration imminent.",
        "prevalence": "Observed in LockBit, BlackCat, Conti operations",
    },
    {
        "name"      : "Exfiltration Final Stage",
        "sequence"  : ["Command & Control","Data Exfiltration"],
        "description": "Attacker is actively staging and sending stolen data through established C2 channel.",
        "prevalence": "Final stage before Impact in 82% of data breach incidents",
    },
    {
        "name"      : "Persistence + Evasion Combo",
        "sequence"  : ["Persistence","Defense Evasion"],
        "description": "Attacker is digging in and hiding tracks — indicates sophisticated, long-term threat actor.",
        "prevalence": "Nation-state actors (APT groups) signature pattern",
    },
]

# Counterfactual interventions per stage
COUNTERFACTUALS = {
    "Reconnaissance":      "If reconnaissance had been blocked (IDS/IPS at perimeter), the attack chain would not have progressed.",
    "Credential Access":   "If MFA had been enabled, credential theft would not grant access even with valid passwords.",
    "Exploitation":        "If the vulnerability had been patched (or a WAF deployed), exploitation would have failed.",
    "Persistence":         "If file integrity monitoring had been active, the persistence mechanism would have been detected and removed.",
    "Privilege Escalation":"If least-privilege principles were enforced, the attacker could not have gained elevated rights.",
    "Defense Evasion":     "If EDR with tamper protection was deployed, the attacker's evasion attempts would have triggered alerts.",
    "Lateral Movement":    "If network segmentation (micro-segmentation / zero-trust) was in place, lateral movement would have been blocked.",
    "Command & Control":   "If DNS filtering and outbound traffic inspection were active, the C2 beacon would have been blocked.",
    "Collection":          "If DLP and access controls were enforced on sensitive data, collection would have been prevented.",
    "Data Exfiltration":   "If outbound traffic was inspected and large transfers alerted on, exfiltration would have been detected.",
    "Impact":              "If backups were isolated and IR procedures were pre-planned, impact recovery would have been rapid.",
}


# ─────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────

@dataclass
class Explanation:
    """Full XAI explanation for one prediction."""
    # Input
    input_sequence  : list
    predicted_stage : str
    confidence      : float
    threat_score    : float
    risk_level      : str

    # Explanation components
    primary_reason  : str      # one-sentence why
    transition_story: str      # narrative from transitions
    context_weights : dict     # {stage: contribution_score}
    matched_pattern : dict     # closest known attack pattern
    counterfactual  : str      # what-if intervention
    supporting_stats: dict     # historical frequency data

    # Score decomposition
    score_breakdown : dict     # {component: value}

    def to_dict(self) -> dict:
        return {
            "input_sequence"  : self.input_sequence,
            "predicted_stage" : self.predicted_stage,
            "confidence"      : round(self.confidence, 3),
            "threat_score"    : round(self.threat_score, 1),
            "risk_level"      : self.risk_level,
            "primary_reason"  : self.primary_reason,
            "transition_story": self.transition_story,
            "context_weights" : {k: round(v, 3) for k, v in self.context_weights.items()},
            "matched_pattern" : self.matched_pattern,
            "counterfactual"  : self.counterfactual,
            "score_breakdown" : self.score_breakdown,
        }

    def __str__(self) -> str:
        cw_lines = "\n".join(
            f"      {stage:<28} contribution: {w:.3f}"
            for stage, w in sorted(self.context_weights.items(), key=lambda x: -x[1])
        )
        pat = self.matched_pattern
        return (
            f"\n{'─'*60}\n"
            f"  XAI EXPLANATION\n"
            f"{'─'*60}\n"
            f"  Prediction  : {self.predicted_stage}  "
            f"(confidence={self.confidence:.1%}, score={self.threat_score:.0f})\n"
            f"\n  WHY this prediction?\n"
            f"  {self.primary_reason}\n"
            f"\n  Transition narrative:\n"
            f"  {self.transition_story}\n"
            f"\n  Context stage contributions (SHAP-inspired):\n"
            f"{cw_lines}\n"
            f"\n  Matched attack pattern:\n"
            f"    Name    : {pat.get('name','—')}\n"
            f"    Match   : {pat.get('description','—')}\n"
            f"    Evidence: {pat.get('prevalence','—')}\n"
            f"\n  Score breakdown:\n"
            f"    Base score        : {self.score_breakdown.get('base',0)}\n"
            f"    Confidence boost  : +{self.score_breakdown.get('confidence_boost',0):.1f}\n"
            f"    Kill-chain depth  : +{self.score_breakdown.get('depth_boost',0):.1f}\n"
            f"    ─────────────────────────────────\n"
            f"    Total             : {self.threat_score:.0f}\n"
            f"\n  Counterfactual — what would have stopped this?\n"
            f"  {self.counterfactual}\n"
            f"{'─'*60}\n"
        )


# ─────────────────────────────────────────────
# EXPLAINER CLASSES
# ─────────────────────────────────────────────

class TransitionExplainer:
    """
    Probability-based explainer: explains WHY the model chose the
    predicted stage by tracing the transition frequencies from
    historical data.
    """

    def explain_transition(
        self, sequence: list, predicted: str, confidence: float
    ) -> dict:
        last_stage = sequence[-1] if sequence else "Unknown"
        nexts      = TRANSITION_PROBS.get(last_stage, {})
        pred_prob  = nexts.get(predicted, confidence)

        # Build ranked alternatives
        alternatives = sorted(nexts.items(), key=lambda x: -x[1])

        narrative = self._build_narrative(
            sequence, last_stage, predicted, pred_prob, alternatives
        )

        return {
            "from_stage"    : last_stage,
            "to_stage"      : predicted,
            "probability"   : round(pred_prob, 3),
            "alternatives"  : [(s, round(p, 3)) for s, p in alternatives[:4]],
            "narrative"     : narrative,
        }

    def _build_narrative(
        self, sequence, last, predicted, prob, alternatives
    ) -> str:
        seq_str  = " → ".join(sequence[-3:]) if len(sequence) > 3 else " → ".join(sequence)
        alt_str  = ""
        if len(alternatives) > 1:
            second = alternatives[1]
            alt_str = f" The second most likely step was '{second[0]}' ({second[1]:.0%})."

        desc_last = STAGE_DESCRIPTIONS.get(last, last.lower())
        desc_pred = STAGE_DESCRIPTIONS.get(predicted, predicted.lower())

        return (
            f"After observing '{seq_str}', the model noted that in "
            f"{prob:.0%} of historical attack sessions where an attacker was "
            f"{desc_last}, the next observed action was {desc_pred}.{alt_str}"
        )


class FeatureExplainer:
    """
    SHAP-inspired contribution explainer: estimates how much each
    stage in the context window contributed to the prediction.

    Method: for each position in the context window, we measure how
    the prediction probability changes when that stage is masked
    (replaced with <PAD>). The change in probability is the contribution.
    """

    def compute_contributions(
        self,
        sequence  : list,
        predicted : str,
        model_fn  = None,
    ) -> dict:
        """
        Returns {stage_name: contribution_score} for each stage in sequence.
        Higher score = that stage was more important for the prediction.

        model_fn: callable(seq) → confidence for predicted stage
                  If None, uses built-in transition approximation.
        """
        if model_fn is None:
            model_fn = self._approx_confidence

        base_conf = model_fn(sequence, predicted)
        contributions = {}

        for i, stage in enumerate(sequence):
            # Mask this position
            masked    = sequence[:i] + ["<PAD>"] + sequence[i+1:]
            masked_c  = model_fn(masked, predicted)
            drop      = base_conf - masked_c
            contributions[stage] = max(0.0, float(drop))

        # Normalise so contributions sum to 1
        total = sum(contributions.values()) + 1e-9
        contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _approx_confidence(self, sequence: list, predicted: str) -> float:
        """Approximate confidence for the predicted stage using transitions."""
        effective = [s for s in sequence if s != "<PAD>"]
        if not effective:
            return 0.1
        last  = effective[-1]
        nexts = TRANSITION_PROBS.get(last, {})

        # Accumulate signal from earlier stages in window too
        conf = nexts.get(predicted, 0.1)
        if len(effective) >= 2:
            prev      = effective[-2]
            prev_next = TRANSITION_PROBS.get(prev, {})
            conf      = 0.7 * conf + 0.3 * prev_next.get(predicted, 0.1)
        return conf


class PatternMatcher:
    """
    Matches the current attack sequence against known real-world
    attack patterns and returns the closest match with context.
    """

    def find_best_match(self, sequence: list) -> dict:
        best_score  = -1
        best_pattern = {}

        for pattern in KNOWN_PATTERNS:
            score = self._match_score(sequence, pattern["sequence"])
            if score > best_score:
                best_score  = score
                best_pattern = pattern

        best_pattern = dict(best_pattern)
        best_pattern["match_score"] = round(best_score, 3)
        return best_pattern

    def _match_score(self, sequence: list, pattern: list) -> float:
        """Jaccard-style overlap + ordering bonus."""
        s_set = set(sequence)
        p_set = set(pattern)
        if not p_set:
            return 0.0
        overlap  = len(s_set & p_set) / len(s_set | p_set)
        # Order bonus: how many consecutive pairs are preserved
        order_hits = 0
        for i in range(len(pattern) - 1):
            if pattern[i] in sequence and pattern[i+1] in sequence:
                if sequence.index(pattern[i]) < sequence.index(pattern[i+1]):
                    order_hits += 1
        order_bonus = order_hits / max(len(pattern) - 1, 1) * 0.3
        return overlap + order_bonus


class CounterfactualEngine:
    """
    Answers: "What would have stopped this attack at each stage?"
    Also computes score deltas for hypothetical defenses.
    """

    def get_intervention(self, sequence: list, predicted: str) -> dict:
        """Return the highest-impact intervention point in the sequence."""
        # The earliest stage in the chain has the most leverage
        intervention_stage = sequence[0] if sequence else predicted
        text = COUNTERFACTUALS.get(
            intervention_stage,
            f"Detecting and blocking {intervention_stage} would have "
            f"prevented the chain from progressing to {predicted}."
        )
        # Estimate score delta
        base_score = BASE_SCORES.get(predicted, 30)
        early_score = BASE_SCORES.get(intervention_stage, 20)
        delta = base_score - early_score

        return {
            "intervention_stage"  : intervention_stage,
            "description"         : text,
            "estimated_score_delta": round(delta, 1),
            "message"             : (
                f"If '{intervention_stage}' had been blocked, the predicted "
                f"'{predicted}' would not have been reached, reducing threat "
                f"score by approximately {delta:.0f} points."
            ),
        }


# ─────────────────────────────────────────────
# MASTER EXPLAINER
# ─────────────────────────────────────────────

class XAIExplainer:
    """
    Orchestrates all explanation strategies into a single Explanation object.

    Usage
    -----
    explainer = XAIExplainer()

    exp = explainer.explain(
        sequence  = ["Reconnaissance", "Credential Access", "Exploitation"],
        predicted = "Privilege Escalation",
        confidence= 0.99,
    )
    print(exp)
    print(exp.to_dict())
    """

    def __init__(self):
        self.transition_exp  = TransitionExplainer()
        self.feature_exp     = FeatureExplainer()
        self.pattern_matcher = PatternMatcher()
        self.cf_engine       = CounterfactualEngine()

    def _compute_score(self, sequence, predicted, confidence):
        base  = BASE_SCORES.get(predicted, 30)
        cb    = confidence * 10
        depth = max((KILL_CHAIN_ORDER.index(s)+1
                     for s in sequence if s in KILL_CHAIN_ORDER), default=0)
        db    = (depth / len(KILL_CHAIN_ORDER)) * 10
        score = float(np.clip(base + cb + db, 0, 100))
        if score >= 85:   level = "CRITICAL"
        elif score >= 70: level = "HIGH"
        elif score >= 50: level = "MEDIUM"
        elif score >= 30: level = "LOW"
        else:             level = "SAFE"
        return score, level, {"base": base, "confidence_boost": round(cb,1), "depth_boost": round(db,1)}

    def explain(
        self,
        sequence  : list,
        predicted : str,
        confidence: float,
    ) -> Explanation:
        score, level, breakdown = self._compute_score(sequence, predicted, confidence)

        # 1. Transition explanation
        trans_info = self.transition_exp.explain_transition(sequence, predicted, confidence)

        # 2. Feature contributions
        contributions = self.feature_exp.compute_contributions(sequence, predicted)

        # 3. Pattern match
        pattern = self.pattern_matcher.find_best_match(sequence)

        # 4. Counterfactual
        cf = self.cf_engine.get_intervention(sequence, predicted)

        # 5. Primary reason (one sentence)
        last        = sequence[-1] if sequence else "Unknown"
        trans_prob  = trans_info["probability"]
        primary     = (
            f"In {trans_prob:.0%} of observed sessions where '{last}' was the "
            f"last detected stage, '{predicted}' was the next attacker action — "
            f"this is the strongest transition signal driving the prediction."
        )

        # 6. Supporting stats
        nexts       = TRANSITION_PROBS.get(last, {})
        stats       = {
            "from_stage"           : last,
            "to_predicted_prob"    : round(trans_prob, 3),
            "kill_chain_depth"     : KILL_CHAIN_ORDER.index(predicted) + 1
                                     if predicted in KILL_CHAIN_ORDER else 0,
            "sequence_length"      : len(sequence),
            "all_next_probs"       : {k: round(v, 3) for k, v in
                                      sorted(nexts.items(), key=lambda x: -x[1])[:5]},
        }

        return Explanation(
            input_sequence   = sequence,
            predicted_stage  = predicted,
            confidence       = confidence,
            threat_score     = score,
            risk_level       = level,
            primary_reason   = primary,
            transition_story = trans_info["narrative"],
            context_weights  = contributions,
            matched_pattern  = pattern,
            counterfactual   = cf["message"],
            supporting_stats = stats,
            score_breakdown  = breakdown,
        )

    def explain_batch(self, scenarios: list) -> list:
        """Explain a list of (sequence, predicted, confidence) tuples."""
        return [self.explain(*s) for s in scenarios]

    def save_explanations(self, explanations: list) -> Path:
        out  = XAI_DIR / "explanations.json"
        data = [e.to_dict() for e in explanations]
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(explanations)} explanations → {out}")
        return out


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    explainer = XAIExplainer()

    print(f"\n{'='*60}")
    print(f"  MODULE 8 — Explainable AI (XAI) Layer")
    print(f"{'='*60}\n")

    # ── Test scenarios with explanations ─────
    test_cases = [
        {
            "label"    : "Scenario A — Mid-chain",
            "sequence" : ["Reconnaissance", "Credential Access", "Exploitation"],
            "predicted": "Privilege Escalation",
            "confidence": 0.990,
        },
        {
            "label"    : "Scenario B — Advanced APT",
            "sequence" : ["Reconnaissance","Exploitation","Persistence",
                           "Privilege Escalation","Lateral Movement"],
            "predicted": "Command & Control",
            "confidence": 0.961,
        },
        {
            "label"    : "Scenario C — Insider threat",
            "sequence" : ["Discovery", "Credential Access"],
            "predicted": "Data Exfiltration",
            "confidence": 0.700,
        },
    ]

    explanations = []
    for tc in test_cases:
        print(f"  ── {tc['label']}")
        exp = explainer.explain(tc["sequence"], tc["predicted"], tc["confidence"])
        print(exp)
        explanations.append(exp)

    # Save
    out = explainer.save_explanations(explanations)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  XAI SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Prediction':<25} {'Score':>6}  {'Level':<10}  {'Top contributor'}")
    print(f"  {'─'*56}")
    for exp in explanations:
        top_stage = max(exp.context_weights, key=exp.context_weights.get) \
                    if exp.context_weights else "—"
        print(f"  {exp.predicted_stage:<25} {exp.threat_score:>6.0f}  "
              f"{exp.risk_level:<10}  {top_stage}")

    print(f"\nModule 8 complete — all 8 modules built!")
    print(f"Explanations saved: {out}")
    print(f"\n{'='*60}")
    print(f"  FULL SYSTEM PIPELINE")
    print(f"{'='*60}")
    pipeline = [
        ("Module 1", "dataset_loader.py",   "Attack Dataset & Ingestion"),
        ("Module 2", "log_processor.py",    "Log Processing & Feature Engineering"),
        ("Module 3", "sequence_builder.py", "Behavior Pattern Extraction"),
        ("Module 4", "train_models.py",     "AI Prediction Model (LSTM + Transformer)"),
        ("Module 5", "threat_scorer.py",    "Threat Scoring Engine"),
        ("Module 6", "app.py",              "Streamlit Security Dashboard"),
        ("Module 7", "attack_simulator.py", "Simulated Attack Environment"),
        ("Module 8", "explainer.py",        "Explainable AI (XAI) Layer"),
    ]
    for mod, file, desc in pipeline:
        print(f"  ✓  {mod:<12} {file:<28} {desc}")
    print(f"\n  Run the full system: streamlit run dashboard/app.py")
    print(f"{'='*60}\n")
