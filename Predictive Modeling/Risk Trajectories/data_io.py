# data_io.py — compact, streamed access to test-patient raw time series.
import pandas as pd
from config import (NMB, OUTCOME_COMPONENTS_TEST, BIOMARKER_COLS, OUT_DIR)

_KEEP = ["uid", "timestamp", "Outcome", "Pupillary Reaction"] + \
        [c for c in BIOMARKER_COLS if c != "Pupillary Reaction"]
_RAW_PARQUET = OUT_DIR / "test_raw.parquet"


def test_uids() -> list:
    s = pd.read_csv(OUTCOME_COMPONENTS_TEST, index_col=0)
    return list(map(str, s.index))


def build_test_raw(uids, out_path=None, chunksize=200_000) -> pd.DataFrame:
    """Stream nmb.csv, keep only `uids`, write+return a compact long table.

    Columns: BIOMARKER_COLS + Outcome; index reset; `timestamp` as float hours.
    """
    out_path = _RAW_PARQUET if out_path is None else out_path
    out_path = _ensure_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = set(map(str, uids))
    cols = set(_KEEP)
    parts = []
    for chunk in pd.read_csv(NMB, index_col=0, chunksize=chunksize):
        chunk = chunk[[c for c in chunk.columns if c in cols]]
        chunk["uid"] = chunk["uid"].astype(str)
        chunk = chunk[chunk["uid"].isin(keep)]
        if len(chunk):
            parts.append(chunk)
    raw = pd.concat(parts, ignore_index=True)
    raw["timestamp"] = pd.to_timedelta(raw["timestamp"]).dt.total_seconds() / 3600.0
    raw = raw.sort_values(["uid", "timestamp"]).reset_index(drop=True)
    raw.to_parquet(out_path)
    return raw


def load_test_raw(path=None) -> pd.DataFrame:
    return pd.read_parquet(_RAW_PARQUET if path is None else path)


def iter_patient_frames(raw: pd.DataFrame, chunk_uids=None):
    """Yield (uid, patient_df) where patient_df is Timedelta-indexed, sorted."""
    uids = list(map(str, chunk_uids)) if chunk_uids is not None else list(raw["uid"].unique())
    sub = raw[raw["uid"].isin(set(uids))]
    for uid, g in sub.groupby("uid", sort=False):
        pdf = g.drop(columns=["uid"]).copy()
        pdf.index = pd.to_timedelta(pdf["timestamp"], unit="h")
        pdf = pdf.drop(columns=["timestamp"]).sort_index()
        yield uid, pdf


def _ensure_path(p):
    """Accept str or Path-like, return pathlib.Path."""
    from pathlib import Path
    return Path(p)
