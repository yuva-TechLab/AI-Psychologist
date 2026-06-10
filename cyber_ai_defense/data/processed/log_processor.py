

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────

# Numeric features used in the ML model
NUMERIC_FEATURES = [
    "dst_port",
    "flow_duration",
    "fwd_pkts",
    "bwd_pkts",
    "fwd_bytes",
    "bwd_bytes",
    "flow_bytes_s",
    "flow_pkts_s",
    # Engineered features (added below)
    "byte_ratio",
    "pkt_ratio",
    "bytes_per_pkt",
    "is_well_known_port",
    "is_ephemeral_port",
    "is_suspicious_port",
]

# Ports commonly associated with lateral movement / C2
SUSPICIOUS_PORTS = {4444, 1337, 9999, 8888, 31337, 6666, 6667, 12345}

# Well-known service ports
WELL_KNOWN_PORTS = set(range(1, 1024))


# ─────────────────────────────────────────────
# PROCESSOR CLASS
# ─────────────────────────────────────────────

class LogProcessor:
    """
    Cleans, engineers, and normalises log data for the AI pipeline.

    Pipeline
    --------
    raw df  →  clean()  →  engineer_features()  →  encode()  →  normalise()  →  ML-ready matrix

    Usage
    -----
    from utils.log_processor import LogProcessor

    proc = LogProcessor()
    df   = proc.load("synthetic_demo.csv")
    df   = proc.clean(df)
    df   = proc.engineer_features(df)
    X, y = proc.encode_and_normalise(df)
    proc.save(df, X, y)
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.label_enc     = LabelEncoder()
        self.imputer       = SimpleImputer(strategy="median")
        self.feature_names = []
        self.fitted        = False

    # ── Load ─────────────────────────────────

    def load(self, filename: str) -> pd.DataFrame:
        path = PROC_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Run Module 1 first. File not found: {path}")
        df = pd.read_csv(path, low_memory=False)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        logger.info(f"Loaded {len(df):,} rows from {filename}")
        return df

    # ── Step 1: Clean ────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates, fix infinities, drop near-zero-variance columns,
        and handle critical missing values.
        """
        original_len = len(df)

        # 1a. Drop exact duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_len - len(df):,} duplicate rows")

        # 1b. Replace inf / -inf with NaN then impute
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # 1c. Cap extreme outliers using IQR (per numeric column)
        for col in ["flow_bytes_s", "flow_pkts_s", "fwd_bytes", "bwd_bytes"]:
            if col in df.columns:
                q1, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(lower=q1, upper=q99)

        # 1d. Fill remaining NaNs with column median
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col]    = df[col].fillna(median_val)

        # 1e. Drop rows where the target label is missing
        if "mitre_stage" in df.columns:
            before = len(df)
            df     = df[df["mitre_stage"].notna() & (df["mitre_stage"] != "")]
            logger.info(f"Dropped {before - len(df):,} rows with missing mitre_stage")

        logger.info(f"Clean: {len(df):,} rows remain (started with {original_len:,})")
        return df.reset_index(drop=True)

    # ── Step 2: Feature Engineering ──────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive meaningful security-relevant features from raw columns.

        New features
        ------------
        byte_ratio          : fwd_bytes / (bwd_bytes + 1)  — asymmetry signals scanning
        pkt_ratio           : fwd_pkts  / (bwd_pkts  + 1)  — asymmetry signals one-way flood
        bytes_per_pkt       : total_bytes / total_pkts      — large = bulk transfer / exfil
        is_well_known_port  : 1 if dst_port < 1024
        is_ephemeral_port   : 1 if dst_port >= 49152
        is_suspicious_port  : 1 if dst_port in known C2/backdoor port list
        hour_of_day         : hour from timestamp (0–23)
        is_off_hours        : 1 if hour outside 08:00–18:00 (attackers love off-hours)
        flow_duration_s     : flow_duration converted to seconds
        """

        # Byte / packet asymmetry
        df["byte_ratio"]    = df["fwd_bytes"] / (df["bwd_bytes"] + 1)
        df["pkt_ratio"]     = df["fwd_pkts"]  / (df["bwd_pkts"]  + 1)

        total_bytes         = df["fwd_bytes"] + df["bwd_bytes"]
        total_pkts          = df["fwd_pkts"]  + df["bwd_pkts"] + 1
        df["bytes_per_pkt"] = total_bytes / total_pkts

        # Port-based flags
        df["is_well_known_port"]  = (df["dst_port"] < 1024).astype(int)
        df["is_ephemeral_port"]   = (df["dst_port"] >= 49152).astype(int)
        df["is_suspicious_port"]  = df["dst_port"].isin(SUSPICIOUS_PORTS).astype(int)

        # Time-based features
        if "timestamp" in df.columns:
            df["hour_of_day"]  = pd.to_datetime(df["timestamp"]).dt.hour
            df["is_off_hours"] = ((df["hour_of_day"] < 8) | (df["hour_of_day"] > 18)).astype(int)
        else:
            df["hour_of_day"]  = 12
            df["is_off_hours"] = 0

        # Flow duration in seconds (CICIDS stores it in microseconds)
        if "flow_duration" in df.columns:
            df["flow_duration_s"] = df["flow_duration"] / 1_000_000
        else:
            df["flow_duration_s"] = 0.0

        # Clip extreme engineered ratios
        df["byte_ratio"]    = df["byte_ratio"].clip(0, 1000)
        df["pkt_ratio"]     = df["pkt_ratio"].clip(0, 1000)
        df["bytes_per_pkt"] = df["bytes_per_pkt"].clip(0, 65535)

        logger.info(f"Engineered features. Total columns: {len(df.columns)}")
        return df

    # ── Step 3: Encode & Normalise ───────────

    def encode_and_normalise(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode the target label and normalise numeric features.

        Returns
        -------
        X : np.ndarray  shape (n, features)  — normalised feature matrix
        y : np.ndarray  shape (n,)           — integer-encoded MITRE stages
        """

        # Extended feature list including engineered columns
        all_numeric = NUMERIC_FEATURES + [
            "hour_of_day", "is_off_hours", "flow_duration_s"
        ]
        available = [c for c in all_numeric if c in df.columns]
        self.feature_names = available
        logger.info(f"Using {len(available)} features: {available}")

        X_raw = df[available].values.astype(np.float32)

        # Impute any remaining NaNs
        if fit:
            X_imp = self.imputer.fit_transform(X_raw)
            X     = self.scaler.fit_transform(X_imp)
            self.fitted = True
        else:
            if not self.fitted:
                raise RuntimeError("Call with fit=True first to fit the scaler.")
            X_imp = self.imputer.transform(X_raw)
            X     = self.scaler.transform(X_imp)

        # Encode target
        if "mitre_stage" in df.columns:
            if fit:
                y = self.label_enc.fit_transform(df["mitre_stage"].values)
            else:
                y = self.label_enc.transform(df["mitre_stage"].values)
        else:
            y = np.zeros(len(df), dtype=int)

        logger.info(f"Encoded: X={X.shape}, y={y.shape}, classes={list(self.label_enc.classes_) if fit else 'reused'}")
        return X, y

    # ── Describe ─────────────────────────────

    def describe(self, df: pd.DataFrame) -> None:
        """Print a feature-level summary after engineering."""
        eng_cols = [
            "byte_ratio", "pkt_ratio", "bytes_per_pkt",
            "is_well_known_port", "is_suspicious_port",
            "is_off_hours", "flow_duration_s"
        ]
        available = [c for c in eng_cols if c in df.columns]

        print("\n" + "=" * 60)
        print("  MODULE 2 — Engineered Feature Summary")
        print("=" * 60)
        print(f"  Total rows    : {len(df):,}")
        print(f"  Total columns : {len(df.columns)}")
        print()
        print(f"  {'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("  " + "-" * 55)
        for col in available:
            if col in df.columns:
                s = df[col]
                print(f"  {col:<25} {s.mean():>10.3f} {s.std():>10.3f} {s.min():>10.3f} {s.max():>10.3f}")

        if "mitre_stage" in df.columns:
            print()
            print("  Stage counts after cleaning:")
            for stage, cnt in df["mitre_stage"].value_counts().items():
                print(f"    {stage:<30} {cnt:>6,}")
        print("=" * 60 + "\n")

    # ── Save ─────────────────────────────────

    def save(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> None:
        """Persist the processed dataframe, feature matrix, and fitted scaler."""

        # Processed CSV
        csv_path = PROC_DIR / "processed_features.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed CSV  → {csv_path}")

        # Numpy arrays
        np.save(PROC_DIR / "X_features.npy", X)
        np.save(PROC_DIR / "y_labels.npy",   y)
        logger.info(f"Saved X ({X.shape}) and y ({y.shape}) arrays")

        # Scaler + encoder metadata
        meta = {
            "feature_names"  : self.feature_names,
            "label_classes"  : list(self.label_enc.classes_),
            "n_features"     : len(self.feature_names),
            "n_classes"      : len(self.label_enc.classes_),
        }
        meta_path = PROC_DIR / "preprocessing_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata       → {meta_path}")

        # Pickle the scaler and encoder for inference time
        with open(PROC_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(PROC_DIR / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_enc, f)
        logger.info("Saved scaler.pkl and label_encoder.pkl")


# ─────────────────────────────────────────────
# STANDALONE DEMO
#   python utils/log_processor.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))

    from data.dataset_loader import DatasetLoader

    # Load synthetic data from Module 1
    loader = DatasetLoader()
    df_raw = loader.load_synthetic(n_samples=5000)

    # Run the full processing pipeline
    proc = LogProcessor()
    df   = proc.clean(df_raw)
    df   = proc.engineer_features(df)
    proc.describe(df)
    X, y = proc.encode_and_normalise(df, fit=True)
    proc.save(df, X, y)

    print(f"\nModule 2 complete.")
    print(f"Feature matrix : X = {X.shape}")
    print(f"Label vector   : y = {y.shape}")
    print(f"Classes        : {list(proc.label_enc.classes_)}")
    print(f"Output dir     : {PROC_DIR}")
    print(f"Next           : Run Module 3 to build attack sequences.\n")
