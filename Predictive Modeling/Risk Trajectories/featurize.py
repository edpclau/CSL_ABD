# featurize.py — window -> 811 catch22 features (faithful replication of training path).
import sys
import numpy as np
import pandas as pd
from config import BIOMARKER_COLS, PUPIL_MAP, N_STEPS, N_FEATURES, HELPERS_DIR

if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))
from HelperFuncsTimeseries import padder   # noqa: E402  (reuse training-time padding)


def prepare_biomarkers(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Return the 45 biomarker columns, ordered, numeric; Pupillary Reaction -> ordinal."""
    df = patient_df.copy()
    df["Pupillary Reaction"] = df["Pupillary Reaction"].map(PUPIL_MAP)
    df = df.reindex(columns=BIOMARKER_COLS)
    return df.apply(pd.to_numeric, errors="coerce")


def pad_window(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Pad an observed window (timestamp-indexed, 45 biomarker cols) to N_STEPS rows.

    Uses the training-time `padder`: a 48-row hourly grid ending at the window's
    latest observed timestamp. Columns preserved/ordered to BIOMARKER_COLS.
    """
    x = obs_df.reset_index()
    if "timestamp" not in x.columns:
        x = x.rename(columns={x.columns[0]: "timestamp"})
    padded = padder(x, pad_length=N_STEPS)          # timestamp-indexed, N_STEPS rows
    padded = padded.reindex(columns=BIOMARKER_COLS)
    if padded.shape[0] != N_STEPS:
        raise ValueError(f"pad_window produced {padded.shape[0]} rows, expected {N_STEPS}")
    return padded
