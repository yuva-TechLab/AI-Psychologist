

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from collections import Counter
from itertools import islice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
SEQ_DIR  = BASE_DIR / "data" / "sequences"
SEQ_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# MITRE ATT&CK KILL-CHAIN ORDER
# ─────────────────────────────────────────────
# Canonical progression used for realistic synthetic sequence generation
# and for ordering stages when timestamps are unavailable.
MITRE_KILL_CHAIN = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command & Control",
    "Exfiltration",
    "Data Exfiltration",
    "Impact",
]

STAGE_TO_IDX = {s: i for i, s in enumerate(MITRE_KILL_CHAIN)}

# Stages present in our synthetic / CICIDS / UNSW data
KNOWN_STAGES = [
    "Benign",
    "Reconnaissance",
    "Credential Access",
    "Exploitation",
    "Privilege Escalation",
    "Lateral Movement",
    "Command & Control",
    "Data Exfiltration",
    "Impact",
    "Persistence",
    "Discovery",
    "Unknown",
]

# Realistic multi-step attack chains used for sequence generation
ATTACK_CHAINS = [
    ["Reconnaissance", "Credential Access", "Exploitation",
     "Privilege Escalation", "Lateral Movement", "Data Exfiltration"],

    ["Reconnaissance", "Exploitation", "Persistence",
     "Command & Control", "Data Exfiltration"],

    ["Reconnaissance", "Credential Access", "Privilege Escalation",
     "Lateral Movement", "Impact"],

    ["Reconnaissance", "Exploitation", "Privilege Escalation",
     "Command & Control", "Data Exfiltration"],

    ["Reconnaissance", "Credential Access", "Lateral Movement",
     "Discovery", "Data Exfiltration"],

    ["Reconnaissance", "Exploitation", "Persistence",
     "Privilege Escalation", "Impact"],
]

# Recommended defensive response per stage
STAGE_DEFENSES = {
    "Reconnaissance"      : "Block port scanning; rate-limit ICMP; enable IDS signatures",
    "Credential Access"   : "Enable MFA; lock accounts after failed attempts; alert on brute force",
    "Exploitation"        : "Patch vulnerable services; enable WAF; deploy honeypots",
    "Privilege Escalation": "Enforce least-privilege; monitor sudo/admin activity; UEBA alerts",
    "Lateral Movement"    : "Segment network; monitor SMB/RDP; disable unnecessary shares",
    "Command & Control"   : "Block suspicious outbound; monitor beaconing; DNS filtering",
    "Data Exfiltration"   : "DLP rules; block large uploads; monitor after-hours transfers",
    "Impact"              : "Isolate affected systems; activate IR plan; restore from backup",
    "Persistence"         : "Audit scheduled tasks/startup; monitor registry; file integrity",
    "Discovery"           : "Alert on whoami/net commands; restrict AD enumeration",
    "Benign"              : "No action required — continue monitoring",
    "Unknown"             : "Investigate manually; increase logging verbosity",
}


# ─────────────────────────────────────────────
# SEQUENCE BUILDER CLASS
# ─────────────────────────────────────────────

class SequenceBuilder:
    """
    Builds attacker behavior sequences from processed log data.

    Usage
    -----
    builder = SequenceBuilder(window_size=3)

    # From a real processed dataframe:
    seqs = builder.build_from_dataframe(df)

    # Or generate realistic synthetic sequences (no data needed):
    seqs = builder.generate_synthetic_sequences(n_sessions=500)

    X_seq, y_seq = builder.to_training_pairs(seqs)
    vocab        = builder.get_vocabulary()
    builder.save(seqs, X_seq, y_seq)
    """

    def __init__(self, window_size: int = 3, min_seq_len: int = 2):
        """
        window_size  : number of past stages used as input context
        min_seq_len  : minimum stages in a session to be included
        """
        self.window_size  = window_size
        self.min_seq_len  = min_seq_len
        self.vocab        = {}          # stage → integer token
        self.inv_vocab    = {}          # integer token → stage
        self.sessions_    = []          # list of stage-name sequences
        self._build_vocab()

    # ── Vocabulary ───────────────────────────

    def _build_vocab(self) -> None:
        """Assign a unique integer token to every known MITRE stage."""
        all_stages = ["<PAD>", "<UNK>"] + KNOWN_STAGES
        self.vocab     = {s: i for i, s in enumerate(all_stages)}
        self.inv_vocab = {i: s for s, i in self.vocab.items()}
        logger.info(f"Vocabulary built: {len(self.vocab)} tokens")

    def encode(self, stage: str) -> int:
        return self.vocab.get(stage, self.vocab["<UNK>"])

    def decode(self, idx: int) -> str:
        return self.inv_vocab.get(idx, "<UNK>")

    # ── Build from real dataframe ─────────────

    def build_from_dataframe(self, df: pd.DataFrame) -> list[list[str]]:
        """
        Group log rows by source IP, sort by timestamp, and extract
        the sequence of MITRE stages for each 'session'.

        Consecutive duplicate stages are collapsed (e.g. Recon → Recon → Cred
        becomes Recon → Cred) to focus on stage *transitions*.
        """
        if "mitre_stage" not in df.columns:
            raise ValueError("DataFrame must have a 'mitre_stage' column (run Module 2 first).")

        group_col = "src_ip" if "src_ip" in df.columns else None
        sort_col  = "timestamp" if "timestamp" in df.columns else None

        sessions = []

        if group_col:
            groups = df.groupby(group_col)
        else:
            # Treat entire dataset as one session if no IP grouping
            groups = [("all", df)]

        for ip, group in groups:
            if sort_col:
                group = group.sort_values(sort_col)

            # Extract stage sequence (exclude benign for attack modeling)
            stages = group["mitre_stage"].tolist()

            # Collapse consecutive duplicates
            deduped = [stages[0]]
            for s in stages[1:]:
                if s != deduped[-1]:
                    deduped.append(s)

            # Keep only sessions with meaningful attack activity
            attack_stages = [s for s in deduped if s != "Benign"]
            if len(attack_stages) >= self.min_seq_len:
                sessions.append(attack_stages)

        self.sessions_ = sessions
        logger.info(f"Extracted {len(sessions):,} attack sessions from {len(df):,} log rows")
        self._print_session_stats(sessions)
        return sessions

    # ── Generate synthetic sequences ─────────

    def generate_synthetic_sequences(
        self,
        n_sessions: int = 500,
        seed: int = 42
    ) -> list[list[str]]:
        """
        Generate realistic multi-step attack sequences based on known
        MITRE ATT&CK kill chains with controlled noise.

        Used when real log data is unavailable (demo / offline mode).
        """
        np.random.seed(seed)
        sessions = []

        for i in range(n_sessions):
            # Pick a base chain
            base = ATTACK_CHAINS[i % len(ATTACK_CHAINS)].copy()

            # Add random noise: occasionally skip a step or insert one
            noise = np.random.random()
            if noise < 0.15 and len(base) > 3:
                # Skip a middle step
                skip_idx = np.random.randint(1, len(base) - 1)
                base.pop(skip_idx)
            elif noise < 0.25:
                # Insert an extra reconnaissance or discovery step
                insert_stage = np.random.choice(["Discovery", "Reconnaissance"])
                insert_idx   = np.random.randint(1, len(base))
                base.insert(insert_idx, insert_stage)

            # Vary session length (partial attacks are common)
            min_len = max(2, len(base) - 2)
            length  = np.random.randint(min_len, len(base) + 1)
            session = base[:length]

            sessions.append(session)

        self.sessions_ = sessions
        logger.info(f"Generated {len(sessions):,} synthetic attack sessions")
        self._print_session_stats(sessions)
        return sessions

    # ── Sliding window → training pairs ──────

    def to_training_pairs(
        self,
        sessions: list[list[str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert sessions into (input_sequence, next_stage) pairs using
        a sliding window of size `window_size`.

        Example (window=3):
          Session: [Recon, Cred, Exploit, PrivEsc, Lateral, Exfil]

          Input                          → Target
          [Recon, Cred, Exploit]         → PrivEsc
          [Cred, Exploit, PrivEsc]       → Lateral
          [Exploit, PrivEsc, Lateral]    → Exfil
        """
        X, y = [], []

        for session in sessions:
            if len(session) < self.window_size + 1:
                # Pad short sessions
                padded  = ["<PAD>"] * (self.window_size + 1 - len(session)) + session
                sessions_aug = [padded]
            else:
                sessions_aug = [session]

            for seq in sessions_aug:
                for i in range(len(seq) - self.window_size):
                    window = seq[i : i + self.window_size]
                    target = seq[i + self.window_size]
                    X.append([self.encode(s) for s in window])
                    y.append(self.encode(target))

        X_arr = np.array(X, dtype=np.int64)
        y_arr = np.array(y, dtype=np.int64)
        logger.info(f"Training pairs: X={X_arr.shape}, y={y_arr.shape}")
        return X_arr, y_arr

    # ── Transition matrix ─────────────────────

    def build_transition_matrix(
        self, sessions: list[list[str]]
    ) -> pd.DataFrame:
        """
        Build a stage-to-stage transition probability matrix.
        Useful for the Markov chain baseline model (Module 4).
        """
        stages   = KNOWN_STAGES
        counts   = pd.DataFrame(0, index=stages, columns=stages)

        for session in sessions:
            for a, b in zip(session[:-1], session[1:]):
                if a in counts.index and b in counts.columns:
                    counts.loc[a, b] += 1

        # Normalise rows → probabilities
        row_sums = counts.sum(axis=1).replace(0, 1)
        probs    = counts.div(row_sums, axis=0)
        return probs

    # ── Stats ─────────────────────────────────

    def _print_session_stats(self, sessions: list[list[str]]) -> None:
        lengths = [len(s) for s in sessions]
        all_stages = [stage for sess in sessions for stage in sess]
        stage_freq = Counter(all_stages)

        print("\n" + "=" * 58)
        print("  MODULE 3 — Sequence Statistics")
        print("=" * 58)
        print(f"  Total sessions       : {len(sessions):,}")
        print(f"  Avg session length   : {np.mean(lengths):.1f} stages")
        print(f"  Min / Max length     : {min(lengths)} / {max(lengths)}")
        print(f"  Total stage tokens   : {len(all_stages):,}")
        print()
        print(f"  {'Stage':<30} {'Count':>7}  Sample sequences")
        print("  " + "-" * 54)
        for stage, cnt in stage_freq.most_common():
            examples = [s for s in sessions if stage in s][:2]
            sample   = " → ".join(examples[0]) if examples else ""
            print(f"  {stage:<30} {cnt:>7,}  {sample[:40]}")
        print("=" * 58 + "\n")

    # ── Helpers ───────────────────────────────

    def get_vocabulary(self) -> dict:
        return {
            "vocab"      : self.vocab,
            "inv_vocab"  : self.inv_vocab,
            "vocab_size" : len(self.vocab),
            "window_size": self.window_size,
            "known_stages": KNOWN_STAGES,
            "defenses"   : STAGE_DEFENSES,
        }

    def decode_sequence(self, token_seq: list[int]) -> list[str]:
        return [self.decode(t) for t in token_seq]

    # ── Save ─────────────────────────────────

    def save(
        self,
        sessions : list[list[str]],
        X        : np.ndarray,
        y        : np.ndarray,
    ) -> None:

        # Save raw sessions as JSON
        sessions_path = SEQ_DIR / "attack_sessions.json"
        with open(sessions_path, "w") as f:
            json.dump(sessions, f, indent=2)

        # Save training arrays
        np.save(SEQ_DIR / "X_sequences.npy", X)
        np.save(SEQ_DIR / "y_targets.npy",   y)

        # Save vocabulary + metadata
        vocab_data = self.get_vocabulary()
        vocab_data["vocab"]     = {k: int(v) for k, v in vocab_data["vocab"].items()}
        vocab_data["inv_vocab"] = {int(k): v for k, v in vocab_data["inv_vocab"].items()}
        with open(SEQ_DIR / "vocabulary.json", "w") as f:
            json.dump(vocab_data, f, indent=2)

        # Save transition matrix
        trans = self.build_transition_matrix(sessions)
        trans.to_csv(SEQ_DIR / "transition_matrix.csv")

        logger.info(f"Saved attack_sessions.json  ({len(sessions):,} sessions)")
        logger.info(f"Saved X_sequences.npy       {X.shape}")
        logger.info(f"Saved y_targets.npy         {y.shape}")
        logger.info(f"Saved vocabulary.json       ({len(self.vocab)} tokens)")
        logger.info(f"Saved transition_matrix.csv")


# ─────────────────────────────────────────────
# STANDALONE DEMO
#   python data/sequence_builder.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    builder = SequenceBuilder(window_size=3)

    # Generate synthetic attack sessions
    sessions = builder.generate_synthetic_sequences(n_sessions=500)

    # Convert to training pairs
    X, y = builder.to_training_pairs(sessions)

    # Show example sequences
    print("  Example attack sequences (first 5):")
    print("  " + "-" * 50)
    for sess in sessions[:5]:
        print(f"  {' → '.join(sess)}")

    print()
    print("  Example training pairs (window=3):")
    print("  " + "-" * 50)
    vocab = builder.get_vocabulary()
    inv   = vocab["inv_vocab"]
    for i in range(min(5, len(X))):
        inp    = " → ".join([inv.get(int(t), "?") for t in X[i]])
        target = inv.get(int(y[i]), "?")
        print(f"  Input: [{inp}]  →  Target: {target}")

    # Transition matrix sample
    print()
    trans = builder.build_transition_matrix(sessions)
    print("  Transition probability matrix (top stages):")
    top = ["Reconnaissance", "Credential Access", "Exploitation",
           "Privilege Escalation", "Lateral Movement"]
    available = [s for s in top if s in trans.index]
    print(trans.loc[available, available].round(3).to_string())

    # Save everything
    builder.save(sessions, X, y)

    print(f"\nModule 3 complete.")
    print(f"Sessions  : {len(sessions):,}")
    print(f"X shape   : {X.shape}")
    print(f"y shape   : {y.shape}")
    print(f"Vocab size: {len(builder.vocab)}")
    print(f"Output dir: {SEQ_DIR}")
    print(f"Next      : Run Module 4 to train the LSTM prediction model.\n")
