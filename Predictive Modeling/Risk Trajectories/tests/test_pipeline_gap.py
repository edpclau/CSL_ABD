# tests/test_pipeline_gap.py
import pandas as pd
import pipeline
from windowing import window_observed
from config import BIOMARKER_COLS


def _gap_patient():
    # early block (0..4h) + late block (200..260h) with a long gap between
    hours = list(range(0, 5)) + list(range(200, 261))
    idx = pd.to_timedelta([f"{h}h" for h in hours])
    df = pd.DataFrame(index=idx)
    df["Outcome"] = 0
    df["Pupillary Reaction"] = "normal"
    for c in BIOMARKER_COLS:
        if c not in df.columns:
            df[c] = 1.0
    return df


def test_windows_for_patient_drops_empty_gap_windows():
    eng = pipeline.Engine()
    df = _gap_patient()
    windows, bios = eng.windows_for_patient(df)
    ks = sorted(w.k for w in windows)
    assert 0 in ks                                       # k=0 is always kept
    # every returned window has at least one observed timepoint
    assert all(window_observed(bios, w.w_start, w.w_end).shape[0] > 0 for w in windows)
    # the gap creates a hole in k -> the kept k values are non-contiguous
    assert (max(ks) - min(ks) + 1) > len(ks)
