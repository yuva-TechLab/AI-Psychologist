"""
Module 7: Simulated Attack Environment
=======================================
Safely simulates cyber attack sequences for testing the prediction
pipeline — inspired by Atomic Red Team & MITRE Caldera concepts.

NO real exploits, payloads, or network activity are generated.
All simulation is purely data/sequence-level for defensive research.

Components
----------
  AttackScenario    — defines a named attack playbook
  AttackSimulator   — executes scenarios step-by-step
  SimulationRunner  — batch-runs scenarios and measures prediction accuracy
  SimulationReport  — generates evaluation metrics

Educational use only — defensive research prototype.
"""

import json
import time
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SEQ_DIR  = BASE_DIR / "data" / "sequences"
SIM_DIR  = BASE_DIR / "simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# ATTACK SCENARIO LIBRARY
# Inspired by MITRE ATT&CK & Atomic Red Team
# No real exploit code — sequence labels only
# ─────────────────────────────────────────────

SCENARIO_LIBRARY = {

    "APT29_Cozy_Bear": {
        "name"       : "APT29 — Cozy Bear Style Intrusion",
        "description": "Nation-state actor using spear-phishing for initial access, then slow lateral movement to avoid detection.",
        "threat_actor": "APT29 (Cozy Bear)",
        "techniques" : ["T1566", "T1078", "T1059", "T1021", "T1041"],
        "stages"     : [
            "Reconnaissance",
            "Credential Access",
            "Exploitation",
            "Persistence",
            "Defense Evasion",
            "Lateral Movement",
            "Command & Control",
            "Data Exfiltration",
        ],
        "noise_level": 0.10,   # 10% chance of skipping/inserting a step
        "speed"      : "slow", # slow APT vs fast ransomware
    },

    "Ransomware_LockBit": {
        "name"       : "LockBit-style Ransomware Attack",
        "description": "Fast-moving ransomware: recon → exploit → escalate → encrypt.",
        "threat_actor": "LockBit 3.0 (cybercriminal)",
        "techniques" : ["T1595", "T1190", "T1548", "T1486"],
        "stages"     : [
            "Reconnaissance",
            "Exploitation",
            "Privilege Escalation",
            "Lateral Movement",
            "Impact",
        ],
        "noise_level": 0.05,
        "speed"      : "fast",
    },

    "Insider_Threat": {
        "name"       : "Malicious Insider Data Theft",
        "description": "Authenticated user abusing legitimate access to exfiltrate sensitive data.",
        "threat_actor": "Malicious insider",
        "techniques" : ["T1078", "T1083", "T1005", "T1048"],
        "stages"     : [
            "Discovery",
            "Credential Access",
            "Collection",
            "Data Exfiltration",
        ],
        "noise_level": 0.15,
        "speed"      : "slow",
    },

    "Supply_Chain_Attack": {
        "name"       : "Supply Chain Compromise (SolarWinds-style)",
        "description": "Attacker compromises a trusted vendor to gain persistent access.",
        "threat_actor": "Nation-state (supply chain)",
        "techniques" : ["T1195", "T1543", "T1021", "T1071"],
        "stages"     : [
            "Reconnaissance",
            "Exploitation",
            "Persistence",
            "Defense Evasion",
            "Command & Control",
            "Lateral Movement",
            "Data Exfiltration",
        ],
        "noise_level": 0.08,
        "speed"      : "slow",
    },

    "Credential_Stuffing": {
        "name"       : "Credential Stuffing → Account Takeover",
        "description": "Automated attack using leaked credentials to take over accounts.",
        "threat_actor": "Cybercriminal botnet",
        "techniques" : ["T1110.004", "T1078", "T1530"],
        "stages"     : [
            "Reconnaissance",
            "Credential Access",
            "Discovery",
            "Data Exfiltration",
        ],
        "noise_level": 0.20,
        "speed"      : "fast",
    },

    "Web_App_Exploit": {
        "name"       : "Web Application Attack Chain",
        "description": "SQL injection → webshell → privilege escalation → pivot.",
        "threat_actor": "Opportunistic attacker",
        "techniques" : ["T1190", "T1505.003", "T1548", "T1021"],
        "stages"     : [
            "Reconnaissance",
            "Exploitation",
            "Persistence",
            "Privilege Escalation",
            "Lateral Movement",
            "Command & Control",
        ],
        "noise_level": 0.12,
        "speed"      : "medium",
    },
}

# Atomic-level actions mapped to each stage
# (descriptive only — no real commands or payloads)
ATOMIC_ACTIONS = {
    "Reconnaissance"      : [
        "Port scan on target subnet (T1595.001)",
        "DNS enumeration of target domain (T1596.001)",
        "OSINT: LinkedIn employee harvesting (T1591)",
        "Shodan query for exposed services (T1595.002)",
    ],
    "Discovery"           : [
        "Network share enumeration (T1135)",
        "Active Directory user enumeration (T1087.002)",
        "Service discovery via net commands (T1007)",
        "File and directory listing (T1083)",
    ],
    "Credential Access"   : [
        "Password spray against Office365 (T1110.003)",
        "NTLM hash capture via Responder (T1557.001)",
        "Kerberoasting service accounts (T1558.003)",
        "Credential dump from LSASS (T1003.001)",
    ],
    "Exploitation"        : [
        "SQL injection on login endpoint (T1190)",
        "CVE exploitation of unpatched service (T1203)",
        "Malicious macro in phishing document (T1566.001)",
        "Log4Shell exploit attempt (T1190)",
    ],
    "Persistence"         : [
        "Scheduled task creation (T1053.005)",
        "Registry Run key modification (T1547.001)",
        "Web shell deployment (T1505.003)",
        "New admin account creation (T1136.001)",
    ],
    "Privilege Escalation": [
        "Token impersonation (T1134.001)",
        "UAC bypass via fodhelper (T1548.002)",
        "Sudo abuse on Linux (T1548.003)",
        "PrintSpoofer local privilege escalation (T1068)",
    ],
    "Defense Evasion"     : [
        "Event log clearing (T1070.001)",
        "AV/EDR process termination (T1562.001)",
        "Timestomping of malicious files (T1070.006)",
        "Living-off-the-land with certutil (T1027)",
    ],
    "Lateral Movement"    : [
        "Pass-the-hash via SMB (T1550.002)",
        "RDP session hijacking (T1563.002)",
        "WMI remote execution (T1047)",
        "SSH lateral movement (T1021.004)",
    ],
    "Command & Control"   : [
        "HTTPS C2 beacon to attacker server (T1071.001)",
        "DNS tunneling for covert channel (T1071.004)",
        "Domain fronting via CDN (T1090.004)",
        "Cobalt Strike Beacon check-in (T1071)",
    ],
    "Collection"          : [
        "Keylogger deployment (T1056.001)",
        "Screen capture of sensitive windows (T1113)",
        "Email collection from Outlook (T1114.001)",
        "Clipboard data harvesting (T1115)",
    ],
    "Data Exfiltration"   : [
        "Compressed archive upload via HTTPS (T1048.002)",
        "FTP exfiltration to attacker server (T1048.003)",
        "Cloud storage upload (T1567.002)",
        "Steganography in image files (T1027.003)",
    ],
    "Impact"              : [
        "Ransomware encryption of file shares (T1486)",
        "MBR wiper deployment (T1561.002)",
        "Database deletion (T1485)",
        "Critical service disruption (T1489)",
    ],
}


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class SimulatedEvent:
    """One atomic step in a simulated attack."""
    step         : int
    stage        : str
    action       : str
    technique_id : str
    timestamp_s  : float
    scenario_name: str

    def __str__(self):
        t = time.strftime("%H:%M:%S", time.gmtime(self.timestamp_s))
        return (f"  [{t}] Step {self.step:02d} | {self.stage:<25} | "
                f"{self.action[:55]}")


@dataclass
class SimulationResult:
    """Full result of running one scenario through the prediction engine."""
    scenario_name   : str
    stages_executed : list
    events          : list
    predictions     : list   # [(true_next, predicted_next, correct, score)]
    accuracy        : float
    avg_threat_score: float
    peak_threat_score: float
    peak_risk_level : str
    duration_steps  : int

    def summary(self) -> str:
        correct = sum(1 for _, _, c, _ in self.predictions if c)
        total   = len(self.predictions)
        return (
            f"\n  {'─'*54}\n"
            f"  Scenario  : {self.scenario_name}\n"
            f"  Steps     : {self.duration_steps}\n"
            f"  Accuracy  : {self.accuracy:.1%}  ({correct}/{total} correct)\n"
            f"  Avg score : {self.avg_threat_score:.1f}\n"
            f"  Peak score: {self.peak_threat_score:.1f}  ({self.peak_risk_level})\n"
            f"  {'─'*54}"
        )


# ─────────────────────────────────────────────
# SIMULATOR
# ─────────────────────────────────────────────

class AttackSimulator:
    """
    Executes named attack scenarios step-by-step and streams events
    to the prediction pipeline for real-time evaluation.

    Usage
    -----
    sim = AttackSimulator()
    events = sim.run_scenario("APT29_Cozy_Bear")

    for event in events:
        print(event)
    """

    def __init__(self, seed: int = 42):
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.default_rng(seed)
        self.scenarios = SCENARIO_LIBRARY

    def list_scenarios(self) -> None:
        print(f"\n{'='*58}")
        print(f"  Available Attack Scenarios")
        print(f"{'='*58}")
        for key, sc in self.scenarios.items():
            print(f"  {key}")
            print(f"    {sc['description']}")
            print(f"    Actor : {sc['threat_actor']}")
            print(f"    Stages: {len(sc['stages'])}  Speed: {sc['speed']}")
            print()
        print(f"{'='*58}\n")

    def run_scenario(
        self,
        scenario_key  : str,
        noise_override: float = None,
        verbose       : bool  = True,
    ) -> list:
        """
        Execute a scenario and return a list of SimulatedEvents.

        noise_override: override scenario's default noise level (0.0–1.0)
        verbose       : print events as they occur
        """
        if scenario_key not in self.scenarios:
            raise KeyError(f"Unknown scenario: {scenario_key}. "
                           f"Available: {list(self.scenarios.keys())}")

        sc    = self.scenarios[scenario_key]
        noise = noise_override if noise_override is not None else sc["noise_level"]

        if verbose:
            print(f"\n{'='*58}")
            print(f"  SIMULATION: {sc['name']}")
            print(f"  Actor     : {sc['threat_actor']}")
            print(f"  Stages    : {len(sc['stages'])}")
            print(f"  Noise     : {noise:.0%}")
            print(f"{'='*58}")

        stages = sc["stages"].copy()

        # Apply noise: randomly skip or duplicate steps
        augmented = []
        for stage in stages:
            if self.rng.random() < noise:
                action = self.rng.choice(["skip", "duplicate", "insert_recon"])
                if action == "skip":
                    continue
                elif action == "duplicate":
                    augmented.append(stage)
                elif action == "insert_recon" and stage != "Reconnaissance":
                    augmented.append("Reconnaissance")
            augmented.append(stage)

        if not augmented:
            augmented = stages   # fallback if all skipped

        # Generate events
        events    = []
        t         = time.time()
        speed_map = {"slow": 3600, "medium": 600, "fast": 60}
        dt        = speed_map.get(sc["speed"], 600)

        for i, stage in enumerate(augmented):
            actions = ATOMIC_ACTIONS.get(stage, [f"Execute {stage} technique"])
            action  = self.rng.choice(actions)
            # Extract technique ID from action string
            tech_id = "T0000"
            if "(" in action and action.endswith(")"):
                tech_id = action.split("(")[-1].rstrip(")")

            event = SimulatedEvent(
                step          = i + 1,
                stage         = stage,
                action        = action.split(" (")[0],
                technique_id  = tech_id,
                timestamp_s   = t + i * dt * self.np_rng.uniform(0.7, 1.3),
                scenario_name = scenario_key,
            )
            events.append(event)

            if verbose:
                print(event)

        if verbose:
            print(f"\n  Simulated {len(events)} events across "
                  f"{len(set(e.stage for e in events))} unique stages.\n")

        return events

    def events_to_sequence(self, events: list) -> list:
        """Convert SimulatedEvents → stage name list (deduplicated)."""
        stages = [e.stage for e in events]
        deduped = [stages[0]]
        for s in stages[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        return deduped


# ─────────────────────────────────────────────
# SIMULATION RUNNER — measures prediction accuracy
# ─────────────────────────────────────────────

class SimulationRunner:
    """
    Runs all scenarios through the prediction model and measures
    how accurately the AI predicted each next step.
    """

    def __init__(self, predictor_fn=None):
        """
        predictor_fn: callable(sequence: list) → (predicted_stage: str, confidence: float)
                      If None, uses the built-in transition-probability predictor.
        """
        self.predictor_fn = predictor_fn or self._default_predictor
        self.simulator    = AttackSimulator()
        self._load_vocab()

    def _load_vocab(self):
        vp = SEQ_DIR / "vocabulary.json"
        if vp.exists():
            with open(vp) as f:
                vdata = json.load(f)
            self.vocab     = {k: int(v) for k, v in vdata["vocab"].items()}
            self.inv_vocab = {int(k): v for k, v in vdata["inv_vocab"].items()}
        else:
            self.vocab = {}; self.inv_vocab = {}

    def _default_predictor(self, sequence: list) -> tuple:
        """Transition-probability predictor (no model files needed)."""
        TRANS = {
            "Reconnaissance":       {"Credential Access": .44, "Exploitation": .36, "Discovery": .12, "Persistence": .08},
            "Credential Access":    {"Exploitation": .34, "Privilege Escalation": .32, "Lateral Movement": .22, "Discovery": .12},
            "Exploitation":         {"Privilege Escalation": .50, "Persistence": .28, "Lateral Movement": .14, "Reconnaissance": .08},
            "Privilege Escalation": {"Lateral Movement": .51, "Discovery": .28, "Command & Control": .14, "Impact": .07},
            "Lateral Movement":     {"Data Exfiltration": .44, "Command & Control": .28, "Discovery": .20, "Impact": .08},
            "Command & Control":    {"Data Exfiltration": .70, "Lateral Movement": .18, "Impact": .12},
            "Discovery":            {"Lateral Movement": .50, "Data Exfiltration": .30, "Impact": .20},
            "Persistence":          {"Privilege Escalation": .40, "Command & Control": .30, "Lateral Movement": .20, "Impact": .10},
            "Data Exfiltration":    {"Impact": .80, "Command & Control": .20},
            "Defense Evasion":      {"Lateral Movement": .50, "Command & Control": .30, "Privilege Escalation": .20},
            "Collection":           {"Data Exfiltration": .70, "Command & Control": .30},
            "Impact":               {"Reconnaissance": .50, "Data Exfiltration": .50},
        }
        last  = sequence[-1] if sequence else "Reconnaissance"
        nexts = TRANS.get(last, {"Unknown": 1.0})
        best  = max(nexts, key=nexts.get)
        return best, nexts[best]

    def _compute_score(self, sequence, predicted, confidence) -> tuple:
        BASE = {"Reconnaissance":20,"Discovery":25,"Credential Access":40,
                "Exploitation":55,"Persistence":55,"Privilege Escalation":65,
                "Defense Evasion":65,"Lateral Movement":70,"Command & Control":75,
                "Data Exfiltration":80,"Impact":80,"Unknown":30,"Collection":70}
        KC   = ["Reconnaissance","Discovery","Credential Access","Exploitation",
                "Persistence","Privilege Escalation","Lateral Movement",
                "Command & Control","Data Exfiltration","Impact"]
        base = BASE.get(predicted, 30)
        cb   = confidence * 10
        depth = max((KC.index(s)+1 for s in sequence if s in KC), default=0)
        db    = (depth / len(KC)) * 10
        score = float(np.clip(base + cb + db, 0, 100))
        if score >= 85:   return score, "CRITICAL"
        elif score >= 70: return score, "HIGH"
        elif score >= 50: return score, "MEDIUM"
        elif score >= 30: return score, "LOW"
        else:             return score, "SAFE"

    def run_all(self, verbose: bool = True) -> list:
        """Run every scenario and collect SimulationResults."""
        all_results = []

        for key in self.simulator.scenarios:
            result = self.evaluate_scenario(key, verbose=verbose)
            all_results.append(result)

        return all_results

    def evaluate_scenario(self, scenario_key: str, verbose: bool = True) -> SimulationResult:
        """Run one scenario and measure step-by-step prediction accuracy."""
        events   = self.simulator.run_scenario(scenario_key, verbose=False)
        sequence = self.simulator.events_to_sequence(events)

        predictions = []
        scores      = []

        for i in range(1, len(sequence)):
            context         = sequence[:i]
            true_next       = sequence[i]
            predicted, conf = self.predictor_fn(context)
            correct         = (predicted.strip() == true_next.strip())
            score, level    = self._compute_score(context, predicted, conf)
            predictions.append((true_next, predicted, correct, score))
            scores.append(score)

            if verbose:
                tick = "✓" if correct else "✗"
                print(f"  {tick} Step {i}: [{' → '.join(context[-2:])}]  "
                      f"→ True: {true_next:<25} Pred: {predicted:<25} "
                      f"Score: {score:.0f}")

        acc       = sum(1 for _, _, c, _ in predictions if c) / len(predictions) if predictions else 0
        peak_pred = predictions[scores.index(max(scores))] if scores else ("—","—",False,0)

        result = SimulationResult(
            scenario_name    = scenario_key,
            stages_executed  = sequence,
            events           = events,
            predictions      = predictions,
            accuracy         = acc,
            avg_threat_score = float(np.mean(scores)) if scores else 0,
            peak_threat_score= float(max(scores)) if scores else 0,
            peak_risk_level  = peak_pred[3] if isinstance(peak_pred[3], str) else
                               self._compute_score([], "", 0)[1],
            duration_steps   = len(sequence),
        )

        if verbose:
            print(result.summary())

        return result

    def generate_report(self, results: list) -> dict:
        """Aggregate metrics across all scenarios."""
        report = {
            "total_scenarios"    : len(results),
            "total_steps"        : sum(r.duration_steps for r in results),
            "overall_accuracy"   : float(np.mean([r.accuracy for r in results])),
            "avg_threat_score"   : float(np.mean([r.avg_threat_score for r in results])),
            "peak_threat_score"  : float(max(r.peak_threat_score for r in results)),
            "scenarios"          : [],
        }
        for r in results:
            report["scenarios"].append({
                "name"       : r.scenario_name,
                "accuracy"   : round(r.accuracy, 4),
                "avg_score"  : round(r.avg_threat_score, 1),
                "peak_score" : round(r.peak_threat_score, 1),
                "peak_level" : r.peak_risk_level,
                "steps"      : r.duration_steps,
            })
        return report

    def save_report(self, report: dict) -> Path:
        out = SIM_DIR / "simulation_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {out}")
        return out


# ─────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    print(f"\n{'='*58}")
    print(f"  MODULE 7 — Simulated Attack Environment")
    print(f"{'='*58}\n")

    sim = AttackSimulator()
    sim.list_scenarios()

    # Demo: run one scenario in verbose mode
    print("Running APT29 scenario (step-by-step):")
    events = sim.run_scenario("APT29_Cozy_Bear", verbose=True)
    sequence = sim.events_to_sequence(events)
    print(f"  Attack sequence: {' → '.join(sequence)}\n")

    # Evaluate all scenarios
    print("="*58)
    print("  Running all scenarios through prediction engine ...")
    print("="*58)
    runner  = SimulationRunner()
    results = runner.run_all(verbose=True)

    # Report
    report = runner.generate_report(results)
    runner.save_report(report)

    print(f"\n{'='*58}")
    print(f"  SIMULATION REPORT")
    print(f"{'='*58}")
    print(f"  Total scenarios  : {report['total_scenarios']}")
    print(f"  Total steps      : {report['total_steps']}")
    print(f"  Overall accuracy : {report['overall_accuracy']:.1%}")
    print(f"  Avg threat score : {report['avg_threat_score']:.1f}")
    print(f"  Peak score seen  : {report['peak_threat_score']:.1f}")
    print()
    print(f"  {'Scenario':<30} {'Acc':>6} {'Avg':>6} {'Peak':>6} {'Level'}")
    print(f"  {'─'*56}")
    for sc in report["scenarios"]:
        print(f"  {sc['name']:<30} {sc['accuracy']:>6.1%} "
              f"{sc['avg_score']:>6.1f} {sc['peak_score']:>6.1f}  {sc['peak_level']}")
    print(f"{'='*58}")
    print(f"\nModule 7 complete. Report saved to simulation/simulation_report.json")
    print(f"Next: Module 8 — Explainable AI (XAI) Layer\n")
