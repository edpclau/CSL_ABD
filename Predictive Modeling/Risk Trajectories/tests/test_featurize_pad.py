# tests/test_featurize_pad.py
import numpy as np
import pandas as pd
import featurize as F
from config import BIOMARKER_COLS, N_STEPS

def test_prepare_biomarkers_maps_pupil_and_orders_cols():
    idx = pd.to_timedelta(["0h", "1h"])
    raw = pd.DataFrame({"Pupillary Reaction": ["normal", "both sluggish"],
                        "Pulse": [100.0, 110.0]}, index=idx)
    # add the rest as NaN so all 45 columns exist
    for c in BIOMARKER_COLS:
        if c not in raw.columns:
            raw[c] = np.nan
    out = F.prepare_biomarkers(raw)
    assert list(out.columns) == BIOMARKER_COLS          # exact order
    assert out["Pupillary Reaction"].tolist() == [0, 2] # ordinal map
    assert pd.api.types.is_numeric_dtype(out["Pupillary Reaction"])

def test_pad_window_returns_48_rows_ending_at_last_obs():
    idx = pd.to_timedelta(["51h", "53h", "55h"])        # gaps within window
    obs = pd.DataFrame({c: np.arange(3.0) for c in BIOMARKER_COLS}, index=idx)
    padded = F.pad_window(obs)
    assert padded.shape == (N_STEPS, len(BIOMARKER_COLS))
    assert list(padded.columns) == BIOMARKER_COLS
    # last row corresponds to the latest observed timestamp (55h)
    assert not padded.iloc[-1].isna().all()
