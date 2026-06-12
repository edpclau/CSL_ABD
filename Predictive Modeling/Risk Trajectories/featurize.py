# featurize.py — window -> 811 catch22 features (faithful replication of training path).
import sys
import warnings
import numpy as np
import pandas as pd
from config import BIOMARKER_COLS, PUPIL_MAP, N_STEPS, N_FEATURES, HELPERS_DIR, SAITS_CKPT

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


class SaitsImputer:
    """Loads the saved SAITS checkpoint once and imputes batches of windows."""

    def __init__(self):
        from pypots.imputation import SAITS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._saits = SAITS(
                n_steps=N_STEPS, n_features=N_FEATURES, n_layers=2, d_model=256,
                d_ffn=128, n_heads=4, d_k=64, d_v=64, dropout=0.1,
                epochs=1, device="cpu",
            )
            self._saits.load(str(SAITS_CKPT))

    def impute_batch(self, arr: np.ndarray) -> np.ndarray:
        """arr: (n_windows, N_STEPS, N_FEATURES) float with NaN -> imputed array same shape."""
        if arr.ndim != 3 or arr.shape[1:] != (N_STEPS, N_FEATURES):
            raise ValueError(f"expected (n,{N_STEPS},{N_FEATURES}), got {arr.shape}")
        return self._saits.impute({"X": arr.astype(np.float32)})
