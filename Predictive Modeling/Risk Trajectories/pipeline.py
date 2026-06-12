# pipeline.py — per-chunk engine: windows -> 811 features -> risk + SHAP.
import numpy as np
import pandas as pd
from config import BIOMARKER_COLS, N_STEPS
from featurize import prepare_biomarkers, pad_window, SaitsImputer, Catch22Featurizer
from windowing import enumerate_windows, window_observed


class Engine:
    """Holds the (heavy) SAITS + catch22 objects; reused across all chunks."""

    def __init__(self):
        self.saits = SaitsImputer()
        self.catch22 = Catch22Featurizer()

    def features_for_windows(self, bios: pd.DataFrame, windows) -> pd.DataFrame:
        """bios: 45-col numeric biomarker frame (timestamp-indexed) for ONE patient.
        windows: list of windowing.Window. Returns (len(windows), 811) feature frame."""
        stack = np.empty((len(windows), N_STEPS, len(BIOMARKER_COLS)), dtype=np.float32)
        for i, w in enumerate(windows):
            obs = window_observed(bios, w.w_start, w.w_end)
            stack[i] = pad_window(obs).to_numpy(dtype=np.float32)
        imputed = self.saits.impute_batch(stack)
        return self.catch22.transform_batch(imputed)

    def windows_for_patient(self, patient_df: pd.DataFrame):
        """Return (windows, bios) for a patient frame (timestamp-indexed, has Outcome).

        Windows with zero observed timepoints are dropped: a patient with a data
        gap can have a 48h window that falls entirely in the gap, which carries no
        signal (and can't be padded/featurized). This never affects k=0 (always
        near the anchor's data), so it leaves the test-set reproduction unchanged.
        """
        bios = prepare_biomarkers(patient_df)
        windows = [w for w in enumerate_windows(patient_df)
                   if window_observed(bios, w.w_start, w.w_end).shape[0] > 0]
        return windows, bios
