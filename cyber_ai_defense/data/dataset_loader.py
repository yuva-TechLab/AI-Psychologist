"""
Module 1: Attack Dataset Loader & Explorer
==========================================
Handles loading and initial exploration of public cybersecurity datasets:
  - CICIDS2017  (simulated network traffic with labeled attacks)
  - UNSW-NB15   (synthetic + real network traffic)
  - Synthetic fallback (for demo / offline use — no download needed)

Educational use only — defensive research prototype.

Download links (optional real data):
  CICIDS2017 : https://www.unb.ca/cic/datasets/ids-2017.html
  UNSW-NB15  : https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
RAW_DIR   = BASE_DIR / "raw"
PROC_DIR  = BASE_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# COLUMN RENAME MAPS
# ─────────────────────────────────────────────
CICIDS_COLS = {
    "Destination Port"           : "dst_port",
    "Flow Duration"              : "flow_duration",
    "Total Fwd Packets"          : "fwd_pkts",
    "Total Backward Packets"     : "bwd_pkts",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Flow Bytes/s"               : "flow_bytes_s",
    "Flow Packets/s"             : "flow_pkts_s",
    "Label"                      : "label",
}

UNSWNB15_COLS = {
    "proto"      : "protocol",
    "service"    : "service",
    "state"      : "state",
    "sbytes"     : "src_bytes",
    "dbytes"     : "dst_bytes",
    "dur"        : "flow_duration",
    "ct_srv_src" : "conn_count",
    "attack_cat" : "label",
    "label"      : "binary_label",
}


# ─────────────────────────────────────────────
# ATTACK LABEL → MITRE ATT&CK STAGE MAPPING
# ─────────────────────────────────────────────
LABEL_TO_STAGE = {
    # CICIDS2017
    "BENIGN"                         : "Benign",
    "PortScan"                       : "Reconnaissance",
    "FTP-Patator"                    : "Credential Access",
    "SSH-Patator"                    : "Credential Access",
    "DoS Hulk"                       : "Impact",
    "DoS GoldenEye"                  : "Impact",
    "DoS slowloris"                  : "Impact",
    "DoS Slowhttptest"               : "Impact",
    "DDoS"                           : "Impact",
    "Heartbleed"                     : "Exploitation",
    "Web Attack – Brute Force"       : "Credential Access",
    "Web Attack – XSS"               : "Exploitation",
    "Web Attack – Sql Injection"     : "Exploitation",
    "Infiltration"                   : "Lateral Movement",
    "Bot"                            : "Command & Control",
    # UNSW-NB15
    "Reconnaissance"                 : "Reconnaissance",
    "Backdoor"                       : "Persistence",
    "DoS"                            : "Impact",
    "Exploits"                       : "Exploitation",
    "Analysis"                       : "Discovery",
    "Fuzzers"                        : "Exploitation",
    "Worms"                          : "Lateral Movement",
    "Shellcode"                      : "Privilege Escalation",
    "Generic"                        : "Exploitation",
    "Normal"                         : "Benign",
}


# ─────────────────────────────────────────────
# LOADER CLASS
# ─────────────────────────────────────────────
class DatasetLoader:
    """
    Loads, standardises, and explores cybersecurity datasets.

    Quick start
    -----------
    loader = DatasetLoader()
    df     = loader.load_synthetic(n_samples=5000)   # no downloads needed
    df     = loader.standardise(df)
    stats  = loader.explore(df)
    loader.save(df, "synthetic_demo.csv")
    """

    def __init__(self):
        self.raw_dir  = RAW_DIR
        self.proc_dir = PROC_DIR

    # ── Real dataset loaders ─────────────────

    def load_cicids(self, filename: str) -> pd.DataFrame:
        """Load a CICIDS2017 CSV from data/raw/."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"\n  File not found: {path}\n"
                f"  Download from: https://www.unb.ca/cic/datasets/ids-2017.html\n"
                f"  Then place the CSV in:  data/raw/{filename}"
            )
        logger.info(f"Loading CICIDS2017 — {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    def load_unswnb15(self, filename: str) -> pd.DataFrame:
        """Load a UNSW-NB15 CSV from data/raw/."""
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"\n  File not found: {path}\n"
                f"  Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
                f"  Then place the CSV in:  data/raw/{filename}"
            )
        logger.info(f"Loading UNSW-NB15 — {path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    # ── Synthetic demo dataset ───────────────

    def load_synthetic(self, n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
        """
        Generate a synthetic dataset that mimics real attack traffic patterns.
        Useful for demos, CI pipelines, and offline development.
        Reflects realistic MITRE ATT&CK stage distributions.
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic dataset ({n_samples:,} samples) ...")

        stages  = [
            "Benign", "Reconnaissance", "Credential Access",
            "Exploitation", "Privilege Escalation", "Lateral Movement",
            "Command & Control", "Data Exfiltration", "Impact",
        ]
        weights = [0.50, 0.15, 0.10, 0.08, 0.05, 0.04, 0.04, 0.02, 0.02]
        labels  = np.random.choice(stages, size=n_samples, p=weights)

        # Simulate plausible feature values per attack stage
        def pick_port(stage):
            if stage == "Reconnaissance":   return np.random.randint(1, 1024)
            if stage == "Credential Access":return int(np.random.choice([21, 22, 23, 80, 443, 3389]))
            if stage == "Data Exfiltration":return int(np.random.choice([443, 8080, 4444, 9999]))
            return int(np.random.randint(1024, 65535))

        dst_ports      = [pick_port(s) for s in labels]
        flow_durations = np.abs(np.random.normal(500_000, 200_000, n_samples))
        fwd_pkts       = np.abs(np.random.poisson(20, n_samples)).astype(int)
        bwd_pkts       = np.abs(np.random.poisson(15, n_samples)).astype(int)
        fwd_bytes      = fwd_pkts * np.random.randint(40, 1500, n_samples)
        bwd_bytes      = bwd_pkts * np.random.randint(40, 1500, n_samples)

        df = pd.DataFrame({
            "timestamp"     : pd.date_range("2024-01-01", periods=n_samples, freq="1s"),
            "src_ip"        : [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(n_samples)],
            "dst_port"      : dst_ports,
            "flow_duration" : flow_durations,
            "fwd_pkts"      : fwd_pkts,
            "bwd_pkts"      : bwd_pkts,
            "fwd_bytes"     : fwd_bytes,
            "bwd_bytes"     : bwd_bytes,
            "flow_bytes_s"  : fwd_bytes / (flow_durations / 1e6 + 1e-9),
            "flow_pkts_s"   : (fwd_pkts + bwd_pkts) / (flow_durations / 1e6 + 1e-9),
            "label"         : labels,
            "mitre_stage"   : labels,   # already in MITRE form for synthetic data
        })

        logger.info(f"Synthetic dataset ready.")
        return df

    # ── Standardise schema ───────────────────

    def standardise(self, df: pd.DataFrame, dataset: str = "synthetic") -> pd.DataFrame:
        """
        Rename columns to the project's common schema and add mitre_stage column.
        dataset: 'cicids' | 'unswnb15' | 'synthetic'
        """
        if dataset == "cicids":
            df = df.rename(columns={k: v for k, v in CICIDS_COLS.items() if k in df.columns})
        elif dataset == "unswnb15":
            df = df.rename(columns={k: v for k, v in UNSWNB15_COLS.items() if k in df.columns})

        # Map raw label → MITRE ATT&CK stage
        if "label" in df.columns and "mitre_stage" not in df.columns:
            df["mitre_stage"] = df["label"].map(
                lambda x: LABEL_TO_STAGE.get(str(x).strip(), "Unknown")
            )

        return df

    # ── Explore ──────────────────────────────

    def explore(self, df: pd.DataFrame) -> dict:
        """Print and return summary statistics for the loaded dataset."""
        total = len(df)

        stage_counts = (
            df["mitre_stage"].value_counts().to_dict()
            if "mitre_stage" in df.columns else {}
        )
        label_counts = (
            df["label"].value_counts().to_dict()
            if "label" in df.columns else {}
        )
        date_range = (
            (str(df["timestamp"].min()), str(df["timestamp"].max()))
            if "timestamp" in df.columns else ("N/A", "N/A")
        )

        stats = {
            "total_records" : total,
            "features"      : list(df.columns),
            "label_counts"  : label_counts,
            "stage_counts"  : stage_counts,
            "missing_values": int(df.isnull().sum().sum()),
            "date_range"    : date_range,
        }

        # Pretty print
        print("\n" + "=" * 58)
        print("  MODULE 1 — Dataset Explorer")
        print("=" * 58)
        print(f"  Total records   : {total:,}")
        print(f"  Feature columns : {len(df.columns)}")
        print(f"  Missing values  : {stats['missing_values']}")
        print(f"  Date range      : {date_range[0]}  →  {date_range[1]}")
        print("\n  MITRE ATT&CK Stage Distribution:")
        print(f"  {'Stage':<30} {'Count':>7}  {'%':>6}  Bar")
        print("  " + "-" * 55)
        for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "█" * max(1, int(pct * 0.8))
            print(f"  {stage:<30} {count:>7,}  {pct:>5.1f}%  {bar}")
        print("=" * 58 + "\n")

        return stats

    # ── Save ─────────────────────────────────

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        """Save standardised dataframe to data/processed/."""
        out = self.proc_dir / filename
        df.to_csv(out, index=False)
        logger.info(f"Saved → {out}  ({len(df):,} rows)")
        return out


# ─────────────────────────────────────────────
# STANDALONE DEMO
#   python data/dataset_loader.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    loader = DatasetLoader()

    print("\n[Demo] Using synthetic dataset (no download required)")
    df    = loader.load_synthetic(n_samples=5000)
    stats = loader.explore(df)
    out   = loader.save(df, "synthetic_demo.csv")

    print(f"\nModule 1 complete.")
    print(f"Output : {out}")
    print(f"Next   : Run Module 2 (log_processor.py) to clean & encode features.\n")
