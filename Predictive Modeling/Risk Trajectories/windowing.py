# windowing.py — per-patient anchor and hourly moving-window enumeration.
from dataclasses import dataclass
import pandas as pd
from config import WINDOW_H, CENSOR_H, STRIDE_H

_SAFETY_MAX_K = 5000   # backstop; no real PICU stay approaches this


@dataclass(frozen=True)
class Window:
    k: int                 # hours-before-anchor of the window end
    w_start: pd.Timedelta
    w_end: pd.Timedelta


def compute_anchor(patient_df: pd.DataFrame) -> pd.Timedelta:
    """Censor timestamp: first Outcome==1 minus censor (case) or last timestamp (control)."""
    is_pos = (patient_df["Outcome"] == 1).to_numpy()
    if is_pos.any():
        first_pos_ts = patient_df.index[is_pos][0]
        return first_pos_ts - pd.Timedelta(hours=CENSOR_H)
    return patient_df.index[-1]


def enumerate_windows(patient_df: pd.DataFrame) -> list[Window]:
    """k=0 (tested window) always emitted; k>=1 while the 48h window stays within the record."""
    anchor = compute_anchor(patient_df)
    first_ts = patient_df.index[0]
    window = pd.Timedelta(hours=WINDOW_H)
    step = pd.Timedelta(hours=STRIDE_H)
    out = []
    k = 0
    while k < _SAFETY_MAX_K:
        w_end = anchor - k * step
        w_start = w_end - window
        if k > 0 and w_start < first_ts:
            break
        out.append(Window(k=k, w_start=w_start, w_end=w_end))
        k += 1
    return out


def window_observed(patient_df: pd.DataFrame, w_start: pd.Timedelta, w_end: pd.Timedelta) -> pd.DataFrame:
    """Rows with w_start <= timestamp < w_end (half-open, matching TemporalDataSubset)."""
    ts = patient_df.index
    mask = (ts >= w_start) & (ts < w_end)
    return patient_df.loc[mask]
