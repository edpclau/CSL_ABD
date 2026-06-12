# Risk & SHAP Trajectories via Moving Window — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every test-set patient, slide the model's 48h observation window backward hour-by-hour through their PICU course, faithfully recompute inputs → ensemble risk → per-biomarker SHAP at each window, and produce risk/SHAP trajectories, per-window AUPRC/AUROC, and figures.

**Architecture:** A streaming, chunked Python pipeline that (per window) replicates the exact training-time feature path — slice 48h → pad to 48 rows → reuse the saved SAITS imputer → catch22 → 811 features — then scores the tuned stacked ensemble (3 XGBoost heads → XGBoost meta) and computes exact TreeSHAP via XGBoost `pred_contribs`. A k=0 validation gate proves the pipeline reproduces the published test set before any full run. Memory is bounded by processing patients in small chunks and streaming SHAP to per-patient `.npz` files.

**Tech Stack:** Python in the `ts_ml` conda env (`/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python`) — pandas, numpy, xgboost 3.2.0, sktime 0.38.5 `Catch22`, pypots 1.0 `SAITS`, pyarrow, scipy, scikit-learn, matplotlib, pytest. **No `shap` package** (XGBoost computes exact TreeSHAP natively).

**Spec:** `docs/superpowers/specs/2026-06-11-risk-shap-trajectories-design.md`

---

## Conventions

- All work lives in `Predictive Modeling/Risk Trajectories/` (the repo root is the `Aim 1.1 Causal Discovery` folder; this plan path is relative to it).
- Run everything with the env python. In a shell:
  ```bash
  export PY="/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python"
  ```
- Run tests from inside `Risk Trajectories/`: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/ -v`
- pypots prints a large ASCII banner on import; ignore it.
- Commit after each task. Branch first: `git checkout -b risk-shap-trajectories` (Task 0).

## Verified facts (do not re-derive; the gate in Task 8 confirms them)

- Raw data: `Deidentified Staged Data/nmb.csv` — `index_col=0` drops a leading unnamed column; then `uid`, `timestamp` (timedelta string), and biomarker/outcome columns. ~2.37M rows, 28,594 patients.
- The 45 biomarker columns, **in this exact order**, equal the unique prefixes of `Control Model/feature_list.txt` and are the positional order SAITS expects. Only `Pupillary Reaction` is non-numeric.
- SAITS checkpoint: `Data Pre-processing/Preprocessing/saits_model/20251023_T144845/SAITS.pypots` (dated 2025-10-24, matching the `c12_w48` files). Architecture: `n_steps=48, n_features=45, n_layers=2, d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64, dropout=0.1`. Load: construct `SAITS(...)` then `.load(path)`; impute: `saits.impute({'X': arr})` returns `(n,48,45)` with no NaN.
- catch22: sktime `from sktime.transformations.panel.catch22 import Catch22`, `Catch22(col_names="str_feat", catch24=True, features=CATCH22_FEATURES)`. Input is a `(instance, time)` MultiIndex panel with biomarker columns; output columns are named `"<biomarker>__<feature>"`. 45 biomarkers × 22 = 990 columns; reindex to the 811 `feature_list.txt` names (order matters).
- Each tuned head and the meta load via `xgboost.XGBClassifier().load_model(json)`. Exact TreeSHAP: `booster.predict(xgb.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)` → `(n, n_features+1)`; `sigmoid(contribs.sum(axis=1)) == predict_proba[:,1]` (atol 1e-5).
- Anchor (mirror `TemporalDataSubset`): case (Outcome ever 1) → timestamp of first `Outcome==1` minus 12h; control → last timestamp. Window k: `[anchor − (k+48)h, anchor − k·h)`, rows `w_start <= ts < w_end`.

---

### Task 0: Branch and scaffold

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/` (folder), `Predictive Modeling/Risk Trajectories/tests/__init__.py`, `Predictive Modeling/Risk Trajectories/__init__.py`

- [ ] **Step 1: Branch**

```bash
cd "/Users/eddie/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Research/Projects/Dissertation/Aim 1/Aim 1.1 Causal Discovery"
git checkout -b risk-shap-trajectories
```

- [ ] **Step 2: Create folders and empty package files**

```bash
mkdir -p "Predictive Modeling/Risk Trajectories/tests"
: > "Predictive Modeling/Risk Trajectories/__init__.py"
: > "Predictive Modeling/Risk Trajectories/tests/__init__.py"
```

- [ ] **Step 3: Verify env**

```bash
export PY="/opt/homebrew/Caskroom/miniconda/base/envs/ts_ml/bin/python"
$PY -c "import xgboost, sktime, pypots, pyarrow, sklearn, matplotlib; print('env OK')" 2>/dev/null
```
Expected: `env OK`

- [ ] **Step 4: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/__init__.py" "Predictive Modeling/Risk Trajectories/tests/__init__.py"
git commit -m "chore: scaffold Risk Trajectories package"
```

---

### Task 1: `config.py` — paths and constants

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/config.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import config

def test_biomarker_cols_match_feature_list_prefixes():
    prefixes, seen = [], set()
    for f in config.FEATURE_LIST_811:
        p = f.rsplit("__", 1)[0]
        if p not in seen:
            seen.add(p); prefixes.append(p)
    assert config.BIOMARKER_COLS == prefixes
    assert len(config.BIOMARKER_COLS) == 45

def test_feature_list_has_811():
    assert len(config.FEATURE_LIST_811) == 811

def test_key_paths_exist():
    assert config.NMB.exists()
    assert config.SAITS_CKPT.exists()
    assert config.FEATURE_LIST.exists()
    assert config.X_TEST_CONTROL.exists()
    assert config.OUTCOME_COMPONENTS_TEST.exists()
    for h in ("EEG", "CT", "MRI"):
        assert (config.ENS_TUNED / f"head_{h}_tuned.json").exists()
    assert (config.ENS_TUNED / "meta_xgb_tuned.json").exists()

def test_catch22_has_22_features():
    assert len(config.CATCH22_FEATURES) == 22
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'config'`)

- [ ] **Step 3: Write `config.py`**

```python
# config.py — paths and constants for the moving-window trajectory pipeline.
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # .../Predictive Modeling/Risk Trajectories
PM = SCRIPT_DIR.parent                                 # .../Predictive Modeling
AIM1 = SCRIPT_DIR.parents[2]                            # .../Aim 1

# --- inputs ---
NMB = AIM1 / "Deidentified Staged Data" / "nmb.csv"
PRE = AIM1 / "Data Pre-processing" / "Preprocessing"
SAITS_CKPT = PRE / "saits_model" / "20251023_T144845" / "SAITS.pypots"
HELPERS_DIR = PRE                                       # for importing HelperFuncsTimeseries.padder

CONTROL = PM / "Control Model"
FEATURE_LIST = CONTROL / "feature_list.txt"
X_TEST_CONTROL = CONTROL / "X_test_control.csv"
OUTCOME_COMPONENTS_TEST = CONTROL / "outcome_components_test.csv"   # has uid index + Outcome/EEG/CT/MRI

ENS = PM / "Ensemble Model"
ENS_TUNED = ENS / "tuned"
ENSEMBLE_PRED = ENS / "ensemble_final_test_predictions.csv"

# --- outputs ---
OUT_DIR = SCRIPT_DIR / "artifacts"
SHAP_DIR = OUT_DIR / "shap"
RISK_PARQUET = OUT_DIR / "risk_trajectories.parquet"
WINDOW_METRICS = OUT_DIR / "window_metrics.csv"
FIG_DIR = SCRIPT_DIR / "figures"

# --- window config (c12_w48) ---
WINDOW_H = 48
CENSOR_H = 12
STRIDE_H = 1
N_STEPS = 48        # SAITS pad length
N_FEATURES = 45

COMPONENTS = ["EEG", "CT", "MRI"]   # head order == meta input column order (predict_ensemble.py)

# columns dropped before featurization (notebook cells 12/14)
CONFOUNDING_COLS = ["elapsed_time", "Outcome_timestamp", "spo2_measure", "picu_los",
                    "min_begin", "max_end", "Ventilator Make/Model"]
BIAS_COLS = ["disch_yr", "race", "sex"]
OUTCOME_COLS = ["BH", "EEG", "CT", "MRI", "Meds", "BHMeds",
                "Haloperidol", "Olanzapine", "Dexmedetomidine", "Outcome"]
DROP_COLS = set(CONFOUNDING_COLS + BIAS_COLS + OUTCOME_COLS + ["arrive_yr", "uid", "timestamp"])

PUPIL_MAP = {"normal": 0, "one sluggish": 1, "both sluggish": 2,
             "one nonreactive": 3, "both nonreactive": 4}

CATCH22_FEATURES = [
    "DN_HistogramMode_5", "DN_HistogramMode_10", "SB_BinaryStats_diff_longstretch0",
    "CO_f1ecac", "CO_FirstMin_ac", "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr", "CO_trev_1_num",
    "CO_HistogramAMI_even_2_5", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40", "SB_BinaryStats_mean_longstretch1", "SB_MotifThree_quantile_hh",
    "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01", "DN_Mean", "DN_Spread_Std",
]

with open(FEATURE_LIST) as _f:
    FEATURE_LIST_811 = [ln.strip() for ln in _f if ln.strip()]

# biomarker columns in feature_list prefix order (== SAITS positional order)
_seen = set()
BIOMARKER_COLS = []
for _feat in FEATURE_LIST_811:
    _p = _feat.rsplit("__", 1)[0]
    if _p not in _seen:
        _seen.add(_p); BIOMARKER_COLS.append(_p)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/config.py" "Predictive Modeling/Risk Trajectories/tests/test_config.py"
git commit -m "feat: config paths and constants for trajectory pipeline"
```

---

### Task 2: `windowing.py` — anchor + hourly window enumeration

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/windowing.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_windowing.py`

The patient frame passed in is indexed by `timestamp` (a `pd.Timedelta`), sorted ascending, with at least an `Outcome` column (0/1 per row).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_windowing.py
import pandas as pd
import windowing as W

def _patient(hours, outcome_at=None):
    idx = pd.to_timedelta([f"{h}h" for h in hours])
    out = [0] * len(hours)
    if outcome_at is not None:
        for i, h in enumerate(hours):
            if h >= outcome_at:
                out[i] = 1
    return pd.DataFrame({"Outcome": out}, index=idx)

def test_anchor_control_is_last_timestamp():
    p = _patient(range(0, 100))           # 0..99h, no outcome
    assert W.compute_anchor(p) == pd.Timedelta("99h")

def test_anchor_case_is_first_positive_minus_censor():
    p = _patient(range(0, 100), outcome_at=80)   # first positive at 80h
    assert W.compute_anchor(p) == pd.Timedelta("68h")   # 80 - 12

def test_control_window_count_and_k0():
    p = _patient(range(0, 100))           # anchor 99h
    ws = W.enumerate_windows(p)
    assert ws[0].k == 0
    assert ws[0].w_end == pd.Timedelta("99h")
    assert ws[0].w_start == pd.Timedelta("51h")
    # k>=1 emitted while w_start >= first_ts (0h): k_max where 99-k-48 >= 0 -> k<=51
    assert max(w.k for w in ws) == 51
    assert ws[-1].w_start == pd.Timedelta("0h")

def test_short_case_only_k0_when_window_underflows():
    # first positive at 50h -> anchor 38h -> k0 window [-10h, 38h): w_start < first_ts
    p = _patient(range(0, 60), outcome_at=50)
    ws = W.enumerate_windows(p)
    assert [w.k for w in ws] == [0]       # only the tested window, padded
    assert ws[0].w_end == pd.Timedelta("38h")

def test_window_observed_is_half_open():
    p = _patient(range(0, 100))
    obs = W.window_observed(p, pd.Timedelta("51h"), pd.Timedelta("99h"))
    assert obs.index.min() == pd.Timedelta("51h")
    assert obs.index.max() == pd.Timedelta("98h")   # 99h excluded (half-open)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_windowing.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'windowing'`)

- [ ] **Step 3: Write `windowing.py`**

```python
# windowing.py — per-patient anchor and hourly moving-window enumeration.
from dataclasses import dataclass
import pandas as pd
from config import WINDOW_H, CENSOR_H, STRIDE_H

_H = pd.Timedelta(hours=1)
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_windowing.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/windowing.py" "Predictive Modeling/Risk Trajectories/tests/test_windowing.py"
git commit -m "feat: per-patient anchor and hourly window enumeration"
```

---

### Task 3: `featurize.py` part 1 — biomarker prep + padding

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/featurize.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_featurize_pad.py`

`pad_window` reuses the **existing** `padder` from `HelperFuncsTimeseries.py` (imported via `config.HELPERS_DIR` on `sys.path`) so padding is byte-for-byte the training-time logic: a 48-row hourly grid ending at the window's latest observed timestamp.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_featurize_pad.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'featurize'`)

- [ ] **Step 3: Write `featurize.py` (part 1)**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_featurize_pad.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/featurize.py" "Predictive Modeling/Risk Trajectories/tests/test_featurize_pad.py"
git commit -m "feat: biomarker prep and training-faithful window padding"
```

---

### Task 4: `featurize.py` part 2 — SAITS imputer wrapper

**Files:**
- Modify: `Predictive Modeling/Risk Trajectories/featurize.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_saits.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_saits.py
import numpy as np
import featurize as F
from config import N_STEPS, N_FEATURES

def test_saits_imputes_batch_no_nan():
    imp = F.SaitsImputer()
    arr = np.random.rand(4, N_STEPS, N_FEATURES).astype(np.float32)
    arr[0, :6, :3] = np.nan
    out = imp.impute_batch(arr)
    assert out.shape == (4, N_STEPS, N_FEATURES)
    assert not np.isnan(out).any()
    # observed (non-missing) entries are preserved
    obs_mask = ~np.isnan(arr)
    assert np.allclose(out[obs_mask], arr[obs_mask], atol=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_saits.py -v`
Expected: FAIL (`AttributeError: module 'featurize' has no attribute 'SaitsImputer'`)

- [ ] **Step 3: Append `SaitsImputer` to `featurize.py`**

```python
import warnings
from config import SAITS_CKPT


class SaitsImputer:
    """Loads the saved SAITS checkpoint once and imputes batches of windows."""

    def __init__(self):
        warnings.filterwarnings("ignore")
        from pypots.imputation import SAITS
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_saits.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/featurize.py" "Predictive Modeling/Risk Trajectories/tests/test_saits.py"
git commit -m "feat: SAITS imputer wrapper reusing saved checkpoint"
```

---

### Task 5: `featurize.py` part 3 — catch22 → 811 features

**Files:**
- Modify: `Predictive Modeling/Risk Trajectories/featurize.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_catch22.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_catch22.py
import numpy as np
import featurize as F
from config import BIOMARKER_COLS, FEATURE_LIST_811, N_STEPS

def test_catch22_outputs_811_named_columns_in_order():
    feat = F.Catch22Featurizer()
    # two windows of imputed (no-NaN) data, shape (n, N_STEPS, 45)
    arr = np.random.rand(2, N_STEPS, len(BIOMARKER_COLS)).astype(np.float32)
    out = feat.transform_batch(arr)
    assert list(out.columns) == FEATURE_LIST_811     # exact 811, exact order
    assert out.shape == (2, 811)
    assert not out.isna().any().any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_catch22.py -v`
Expected: FAIL (`AttributeError: ... 'Catch22Featurizer'`)

- [ ] **Step 3: Append `Catch22Featurizer` to `featurize.py`**

```python
from config import CATCH22_FEATURES, FEATURE_LIST_811


class Catch22Featurizer:
    """Batch catch22 over windows -> 811-column frame (feature_list order)."""

    def __init__(self):
        from sktime.transformations.panel.catch22 import Catch22
        self._tr = Catch22(col_names="str_feat", catch24=True, features=CATCH22_FEATURES)

    def transform_batch(self, arr: np.ndarray) -> pd.DataFrame:
        """arr: (n_windows, N_STEPS, 45) imputed -> DataFrame (n_windows, 811)."""
        n = arr.shape[0]
        # build a (instance, time) MultiIndex panel with biomarker columns
        flat = arr.reshape(n * N_STEPS, len(BIOMARKER_COLS))
        idx = pd.MultiIndex.from_product([range(n), range(N_STEPS)], names=["instance", "time"])
        panel = pd.DataFrame(flat, index=idx, columns=BIOMARKER_COLS)
        feats = self._tr.fit_transform(panel)               # (n, 990) named "<bio>__<feat>"
        feats = feats.reindex(columns=FEATURE_LIST_811)     # select + order the 811
        return feats
```

Note: any catch22 column that comes out non-finite (e.g. a degenerate constant series) is left as-is — XGBoost handles NaN natively, and the gate in Task 8 confirms the values match the saved features. The test uses random data so all 811 are finite.

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_catch22.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/featurize.py" "Predictive Modeling/Risk Trajectories/tests/test_catch22.py"
git commit -m "feat: catch22 featurizer to 811 ordered features"
```

---

### Task 6: `model.py` — load, predict, TreeSHAP

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/model.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import numpy as np
import pandas as pd
from scipy.special import expit
import model as M
from config import FEATURE_LIST_811, COMPONENTS

def test_shap_additivity_per_head():
    bundle = M.load_models()
    X = pd.DataFrame(np.random.rand(5, 811).astype(np.float32), columns=FEATURE_LIST_811)
    comps = M.predict_components(X, bundle)           # (5,3)
    shap = M.head_shap(X, bundle)                     # dict head -> (5, 812)
    for j, h in enumerate(COMPONENTS):
        recon = expit(shap[h].sum(axis=1))
        assert np.allclose(recon, comps[:, j], atol=1e-5)

def test_ensemble_risk_in_unit_interval():
    bundle = M.load_models()
    X = pd.DataFrame(np.random.rand(5, 811).astype(np.float32), columns=FEATURE_LIST_811)
    comps = M.predict_components(X, bundle)
    risk = M.predict_ensemble(comps, bundle)
    assert risk.shape == (5,)
    assert ((risk >= 0) & (risk <= 1)).all()

def test_meta_shap_additivity():
    bundle = M.load_models()
    comps = np.random.rand(4, 3).astype(np.float32)
    mshap = M.meta_shap(comps, bundle)                # (4,4)
    risk = M.predict_ensemble(comps, bundle)
    assert np.allclose(expit(mshap.sum(axis=1)), risk, atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_model.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'model'`)

- [ ] **Step 3: Write `model.py`**

```python
# model.py — load the tuned ensemble; ensemble risk + exact TreeSHAP.
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from config import ENS_TUNED, COMPONENTS


@dataclass
class Bundle:
    heads: dict          # component -> XGBClassifier
    meta: object         # XGBClassifier
    feat_names: list     # 811 head feature names (from booster)


def load_models() -> Bundle:
    heads = {}
    for c in COMPONENTS:
        m = xgb.XGBClassifier()
        m.load_model(str(ENS_TUNED / f"head_{c}_tuned.json"))
        heads[c] = m
    meta = xgb.XGBClassifier()
    meta.load_model(str(ENS_TUNED / "meta_xgb_tuned.json"))
    feat_names = heads[COMPONENTS[0]].get_booster().feature_names
    return Bundle(heads=heads, meta=meta, feat_names=feat_names)


def predict_components(X811: pd.DataFrame, b: Bundle) -> np.ndarray:
    """(n,3) component probabilities in COMPONENTS order."""
    return np.column_stack([b.heads[c].predict_proba(X811)[:, 1] for c in COMPONENTS])


def predict_ensemble(components: np.ndarray, b: Bundle) -> np.ndarray:
    """(n,) ensemble P(Outcome) from the 3 component probabilities."""
    return b.meta.predict_proba(np.asarray(components, dtype=np.float32))[:, 1]


def head_shap(X811: pd.DataFrame, b: Bundle) -> dict:
    """Exact TreeSHAP per head vs the 811 biomarker features: head -> (n, 812)."""
    out = {}
    for c in COMPONENTS:
        booster = b.heads[c].get_booster()
        d = xgb.DMatrix(X811.to_numpy(dtype=np.float32), feature_names=booster.feature_names)
        out[c] = booster.predict(d, pred_contribs=True)     # (n, 812)
    return out


def meta_shap(components: np.ndarray, b: Bundle) -> np.ndarray:
    """Exact TreeSHAP of the meta-learner vs the 3 components: (n, 4)."""
    booster = b.meta.get_booster()
    d = xgb.DMatrix(np.asarray(components, dtype=np.float32), feature_names=booster.feature_names)
    return booster.predict(d, pred_contribs=True)           # (n, 4) = 3 comps + bias
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/model.py" "Predictive Modeling/Risk Trajectories/tests/test_model.py"
git commit -m "feat: ensemble loading, prediction, and exact TreeSHAP"
```

---

### Task 7: `data_io.py` — build compact test-patient raw table

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/data_io.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_data_io.py`

Streams `nmb.csv` in chunks (never holding the full 605 MB), keeps only the 3,895 test uids, and writes `artifacts/test_raw.parquet` (biomarker columns + `Outcome` + `timestamp` as hours). Downstream code loads only this compact table.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_io.py
import pandas as pd
import data_io as D
from config import BIOMARKER_COLS

def test_test_uids_loads_3895():
    uids = D.test_uids()
    assert len(uids) == 3895
    assert all(isinstance(u, str) for u in uids[:3])

def test_iter_patient_frames_shape(tmp_path):
    # build on a tiny uid subset to keep the test fast
    uids = D.test_uids()[:3]
    raw = D.build_test_raw(uids, out_path=tmp_path / "mini_raw.parquet")
    seen = 0
    for uid, pdf in D.iter_patient_frames(raw, chunk_uids=uids):
        assert "Outcome" in pdf.columns
        assert set(BIOMARKER_COLS).issubset(pdf.columns)
        assert pdf.index.is_monotonic_increasing
        seen += 1
    assert seen == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_data_io.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'data_io'`)

- [ ] **Step 3: Write `data_io.py`**

```python
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
    """Yield (uid, patient_df) where patient_df is timestamp(Timedelta)-indexed, sorted."""
    uids = list(map(str, chunk_uids)) if chunk_uids is not None else list(raw["uid"].unique())
    sub = raw[raw["uid"].isin(set(uids))]
    for uid, g in sub.groupby("uid", sort=False):
        pdf = g.drop(columns=["uid"]).copy()
        pdf.index = pd.to_timedelta(pdf["timestamp"], unit="h")
        pdf = pdf.drop(columns=["timestamp"]).sort_index()
        yield uid, pdf
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_data_io.py -v`
Expected: PASS (2 tests; the second builds a 3-patient parquet)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/data_io.py" "Predictive Modeling/Risk Trajectories/tests/test_data_io.py"
git commit -m "feat: streamed compact test-patient raw table"
```

---

### Task 8: `pipeline.py` + validation gate (k=0 reproduces the saved test set)

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/pipeline.py` (the per-chunk window→features→risk→shap engine)
- Create: `Predictive Modeling/Risk Trajectories/validate_k0.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_validate_k0.py`

`pipeline.featurize_windows(...)` turns a list of `(uid, Window, observed_df)` into the 811-feature matrix; `validate_k0` computes k=0 for a sample of test patients and asserts the features match `X_test_control.csv` and the risk matches `ensemble_final_test_predictions.csv`.

**This is the gate. Do not proceed to Task 9 until it passes.**

- [ ] **Step 1: Write `pipeline.py`**

```python
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
        """Return (windows, bios) for a patient frame (timestamp-indexed, has Outcome)."""
        windows = enumerate_windows(patient_df)
        bios = prepare_biomarkers(patient_df)
        return windows, bios
```

- [ ] **Step 2: Write `validate_k0.py`**

```python
# validate_k0.py — prove the moving-window pipeline reproduces the saved test set at k=0.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import data_io
from config import X_TEST_CONTROL, ENSEMBLE_PRED, FEATURE_LIST_811
from pipeline import Engine
from model import load_models, predict_components, predict_ensemble


def run(n_patients=60, feat_atol=1e-3, risk_atol=5e-3):
    uids = data_io.test_uids()[:n_patients]
    raw = data_io.build_test_raw(uids)                       # compact table for the sample
    Xref = pd.read_csv(X_TEST_CONTROL, index_col=0)
    pref = pd.read_csv(ENSEMBLE_PRED, index_col=0)

    eng = Engine()
    bundle = load_models()

    feat_rows, risk_rows, ord_uids = [], [], []
    for uid, pdf in data_io.iter_patient_frames(raw, chunk_uids=uids):
        windows, bios = eng.windows_for_patient(pdf)
        w0 = [w for w in windows if w.k == 0]
        X = eng.features_for_windows(bios, w0)               # (1, 811)
        comps = predict_components(X, bundle)
        risk = predict_ensemble(comps, bundle)
        feat_rows.append(X.iloc[0].to_numpy()); risk_rows.append(risk[0]); ord_uids.append(uid)

    Xk0 = pd.DataFrame(feat_rows, index=ord_uids, columns=FEATURE_LIST_811)
    risk = pd.Series(risk_rows, index=ord_uids)

    # feature agreement
    common = [u for u in ord_uids if u in Xref.index]
    diff = (Xk0.loc[common] - Xref.loc[common, FEATURE_LIST_811]).abs()
    feat_max = np.nanmax(diff.to_numpy())
    feat_corr = np.corrcoef(Xk0.loc[common].to_numpy().ravel(),
                            Xref.loc[common, FEATURE_LIST_811].to_numpy().ravel())[0, 1]
    # risk agreement
    risk_max = (risk.loc[common] - pref.loc[common, "p_outcome"]).abs().max()
    print(f"feat max|Δ|={feat_max:.5f}  feat corr={feat_corr:.6f}  risk max|Δ|={risk_max:.5f}")
    print(f"k0 AUPRC={average_precision_score(pref.loc[common,'Outcome'], risk.loc[common]):.4f} "
          f"AUROC={roc_auc_score(pref.loc[common,'Outcome'], risk.loc[common]):.4f}")

    assert feat_corr > 0.999, f"feature correlation too low: {feat_corr}"
    assert feat_max < feat_atol, f"feature max abs diff too high: {feat_max}"
    assert risk_max < risk_atol, f"risk max abs diff too high: {risk_max}"
    return feat_max, feat_corr, risk_max


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Write the gate test**

```python
# tests/test_validate_k0.py
import validate_k0

def test_k0_reproduces_saved_features_and_risk():
    feat_max, feat_corr, risk_max = validate_k0.run(n_patients=40)
    assert feat_corr > 0.999
    assert risk_max < 5e-3
```

- [ ] **Step 4: Run the gate**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_validate_k0.py -v -s`
Expected: PASS. The printed line should show `feat corr` ≈ 0.999+ and small `risk max|Δ|`.

**If it FAILS, stop and diagnose before continuing (spec §6 / §12):**
- Try the other SAITS checkpoint `20251021_T131941` (set `SAITS_CKPT` in `config.py`); re-run.
- Print the worst-disagreeing feature columns: are they one biomarker (mapping issue) or broad (SAITS/scale issue)?
- Confirm `timestamp` rounding: if `padder` emits >48 rows for some patient, raw timestamps aren't hourly-aligned — round `raw["timestamp"]` to the nearest hour in `data_io.build_test_raw` and re-run.
- Do NOT loosen the tolerances to force a pass.

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/pipeline.py" "Predictive Modeling/Risk Trajectories/validate_k0.py" "Predictive Modeling/Risk Trajectories/tests/test_validate_k0.py"
git commit -m "feat: window->features->risk engine and k=0 validation gate"
```

---

### Task 9: `compute_trajectories.py` — streaming driver writing artifacts

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/compute_trajectories.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_compute_smoke.py`

Processes patients in chunks; per patient writes `artifacts/shap/<uid>.npz` (full per-head SHAP) and appends rows to `artifacts/risk_trajectories.parquet`. Feature names are written once to `artifacts/shap/feature_names.json`.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_compute_smoke.py
import json
import numpy as np
import pandas as pd
import compute_trajectories as C
import data_io
from config import FEATURE_LIST_811

def test_compute_small_subset(tmp_path):
    uids = data_io.test_uids()[:5]
    out = C.run(uids=uids, chunk_size=2, out_dir=tmp_path)
    risk = pd.read_parquet(out["risk_parquet"])
    assert {"uid", "k", "ensemble_risk", "EEG_p", "CT_p", "MRI_p", "y_true", "n_observed"} <= set(risk.columns)
    assert (risk["k"] == 0).sum() == 5                      # every patient has k=0
    assert ((risk["ensemble_risk"] >= 0) & (risk["ensemble_risk"] <= 1)).all()
    # one shap file per patient, shape (n_windows, 812) per head
    u0 = uids[0]
    z = np.load(out["shap_dir"] / f"{u0}.npz")
    n_w = (risk["uid"] == u0).sum()
    assert z["eeg"].shape == (n_w, 812)
    assert z["meta"].shape == (n_w, 4)
    assert z["k"].shape == (n_w,)
    names = json.loads((out["shap_dir"] / "feature_names.json").read_text())
    assert names == FEATURE_LIST_811
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_compute_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'compute_trajectories'`)

- [ ] **Step 3: Write `compute_trajectories.py`**

```python
# compute_trajectories.py — streaming, chunked driver. Writes risk parquet + per-patient SHAP.
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import data_io
from config import (OUT_DIR, SHAP_DIR, RISK_PARQUET, FEATURE_LIST_811, COMPONENTS)
from pipeline import Engine
from model import load_models, predict_components, predict_ensemble, head_shap, meta_shap

_RISK_COLS = ["uid", "k", "n_observed", "EEG_p", "CT_p", "MRI_p", "ensemble_risk", "y_true"]


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run(uids=None, chunk_size=50, out_dir=None, log_every=1):
    out_dir = OUT_DIR if out_dir is None else out_dir
    shap_dir = out_dir / "shap"
    risk_parquet = out_dir / "risk_trajectories.parquet"
    shap_dir.mkdir(parents=True, exist_ok=True)
    (shap_dir / "feature_names.json").write_text(json.dumps(FEATURE_LIST_811))

    all_uids = data_io.test_uids() if uids is None else list(map(str, uids))
    raw = data_io.build_test_raw(all_uids)
    ytrue = pd.read_csv(__import__("config").OUTCOME_COMPONENTS_TEST, index_col=0)["Outcome"].astype(int)
    ytrue.index = ytrue.index.map(str)

    eng = Engine()
    bundle = load_models()
    writer = None
    try:
        for ci, chunk in enumerate(_chunks(all_uids, chunk_size)):
            for uid, pdf in data_io.iter_patient_frames(raw, chunk_uids=chunk):
                windows, bios = eng.windows_for_patient(pdf)
                X = eng.features_for_windows(bios, windows)          # (n_w, 811)
                comps = predict_components(X, bundle)               # (n_w, 3)
                risk = predict_ensemble(comps, bundle)              # (n_w,)
                hs = head_shap(X, bundle)                           # head -> (n_w, 812)
                ms = meta_shap(comps, bundle)                      # (n_w, 4)
                ks = np.array([w.k for w in windows], dtype=np.int32)
                n_obs = np.array(
                    [int(((bios.index >= w.w_start) & (bios.index < w.w_end)).sum()) for w in windows],
                    dtype=np.int32)

                np.savez_compressed(
                    shap_dir / f"{uid}.npz",
                    eeg=hs["EEG"].astype(np.float32), ct=hs["CT"].astype(np.float32),
                    mri=hs["MRI"].astype(np.float32), meta=ms.astype(np.float32), k=ks)

                rows = pd.DataFrame({
                    "uid": uid, "k": ks, "n_observed": n_obs,
                    "EEG_p": comps[:, 0], "CT_p": comps[:, 1], "MRI_p": comps[:, 2],
                    "ensemble_risk": risk, "y_true": int(ytrue.get(uid, -1)),
                })[_RISK_COLS]
                table = pa.Table.from_pandas(rows, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(risk_parquet, table.schema)
                writer.write_table(table)
            if ci % log_every == 0:
                print(f"chunk {ci+1}: through {min((ci+1)*chunk_size, len(all_uids))}/{len(all_uids)} patients", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return {"risk_parquet": risk_parquet, "shap_dir": shap_dir}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N test patients")
    ap.add_argument("--chunk-size", type=int, default=50)
    args = ap.parse_args()
    uids = None if args.limit is None else data_io.test_uids()[:args.limit]
    out = run(uids=uids, chunk_size=args.chunk_size)
    print("done ->", out["risk_parquet"])
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_compute_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/compute_trajectories.py" "Predictive Modeling/Risk Trajectories/tests/test_compute_smoke.py"
git commit -m "feat: streaming chunked trajectory driver writing risk + SHAP artifacts"
```

---

### Task 10: `window_metrics.py` — per-window AUPRC/AUROC + headline average

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/window_metrics.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_window_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_window_metrics.py
import numpy as np
import pandas as pd
import window_metrics as WM

def test_metrics_per_offset_and_overall():
    # 3 cases, 3 controls, each with windows at k=0,1; risk separates classes
    rows = []
    for i in range(6):
        y = 1 if i < 3 else 0
        for k in (0, 1):
            rows.append({"uid": f"u{i}", "k": k, "y_true": y,
                         "ensemble_risk": 0.9 - 0.05 * k if y else 0.1 + 0.05 * k})
    risk = pd.DataFrame(rows)
    per_k, overall = WM.compute(risk)
    assert set(per_k["k"]) == {0, 1}
    assert (per_k["n_patients"] == 6).all()
    assert (per_k["AUPRC"] > 0.9).all()
    assert "auprc_pooled" in overall and "auprc_macro" in overall
    assert 0.9 < overall["auprc_pooled"] <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_window_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `window_metrics.py`**

```python
# window_metrics.py — discrimination as a function of lead time + headline averages.
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from config import RISK_PARQUET, WINDOW_METRICS


def compute(risk: pd.DataFrame):
    """Return (per_k DataFrame[k,n_patients,n_pos,AUPRC,AUROC], overall dict)."""
    per = []
    for k, g in risk.groupby("k"):
        y, p = g["y_true"].to_numpy(), g["ensemble_risk"].to_numpy()
        npos = int((y == 1).sum())
        auprc = average_precision_score(y, p) if 0 < npos < len(y) else np.nan
        auroc = roc_auc_score(y, p) if 0 < npos < len(y) else np.nan
        per.append({"k": int(k), "n_patients": len(g), "n_pos": npos,
                    "AUPRC": auprc, "AUROC": auroc})
    per_k = pd.DataFrame(per).sort_values("k").reset_index(drop=True)
    overall = {
        "auprc_pooled": float(average_precision_score(risk["y_true"], risk["ensemble_risk"])),
        "auroc_pooled": float(roc_auc_score(risk["y_true"], risk["ensemble_risk"])),
        "auprc_macro": float(per_k["AUPRC"].mean(skipna=True)),
        "auroc_macro": float(per_k["AUROC"].mean(skipna=True)),
        "n_window_rows": int(len(risk)),
    }
    return per_k, overall


def run(risk_parquet=None, out_csv=None):
    risk = pd.read_parquet(RISK_PARQUET if risk_parquet is None else risk_parquet)
    per_k, overall = compute(risk)
    out_csv = WINDOW_METRICS if out_csv is None else out_csv
    per_k.to_csv(out_csv, index=False)
    print("Headline:", {k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()})
    print(f"k=0 AUPRC={per_k.loc[per_k.k==0,'AUPRC'].iloc[0]:.4f} "
          f"AUROC={per_k.loc[per_k.k==0,'AUROC'].iloc[0]:.4f} (expect ~0.7913 / 0.9229)")
    return per_k, overall


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_window_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/window_metrics.py" "Predictive Modeling/Risk Trajectories/tests/test_window_metrics.py"
git commit -m "feat: per-window AUPRC/AUROC and headline averages"
```

---

### Task 11: `make_figures.py` — trajectory charts

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/make_figures.py`
- Test: `Predictive Modeling/Risk Trajectories/tests/test_figures_smoke.py`

Reads artifacts (one patient's SHAP at a time) and writes PDFs to `figures/`. Functions are individually testable; a smoke test runs them on the small-subset artifacts.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_figures_smoke.py
import pandas as pd
import compute_trajectories as C
import data_io, make_figures as MF

def test_figures_render(tmp_path):
    uids = data_io.test_uids()[:6]
    out = C.run(uids=uids, chunk_size=3, out_dir=tmp_path)
    risk = pd.read_parquet(out["risk_parquet"])
    fig_dir = tmp_path / "figures"
    MF.fig_aggregate_risk(risk, fig_dir)
    MF.fig_window_metrics(risk, fig_dir)
    MF.fig_examples(risk, out["shap_dir"], fig_dir, n_per_class=1)
    assert (fig_dir / "aggregate_risk_vs_lead.pdf").exists()
    assert (fig_dir / "discrimination_vs_lead.pdf").exists()
    assert any(fig_dir.glob("example_*.pdf"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_figures_smoke.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write `make_figures.py`**

```python
# make_figures.py — risk/SHAP trajectory figures from artifacts.
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from window_metrics import compute
from config import FIG_DIR, COMPONENTS


def _ensure(fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def fig_aggregate_risk(risk: pd.DataFrame, fig_dir=FIG_DIR):
    fig_dir = _ensure(fig_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, label, color in [(1, "Case (Outcome=1)", "crimson"), (0, "Control", "steelblue")]:
        g = risk[risk["y_true"] == y]
        stat = g.groupby("k")["ensemble_risk"].agg(["mean", "count",
              lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        stat.columns = ["mean", "count", "q25", "q75"]
        stat = stat[stat["count"] >= 10]
        ax.plot(stat.index, stat["mean"], color=color, label=label)
        ax.fill_between(stat.index, stat["q25"], stat["q75"], color=color, alpha=0.15)
    ax.set_xlabel("hours before anchor (k; 0 = tested window)")
    ax.set_ylabel("mean ensemble risk")
    ax.set_title("Risk trajectory vs lead time")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "aggregate_risk_vs_lead.pdf"); plt.close(fig)


def fig_window_metrics(risk: pd.DataFrame, fig_dir=FIG_DIR):
    fig_dir = _ensure(fig_dir)
    per_k, _ = compute(risk)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(per_k["k"], per_k["AUPRC"], color="darkgreen", label="AUPRC")
    ax1.plot(per_k["k"], per_k["AUROC"], color="darkorange", label="AUROC")
    ax1.set_xlabel("hours before anchor (k)")
    ax1.set_ylabel("discrimination")
    ax2 = ax1.twinx()
    ax2.fill_between(per_k["k"], per_k["n_patients"], color="gray", alpha=0.12)
    ax2.set_ylabel("n patients at offset (length bias →)")
    ax1.legend(loc="lower left")
    ax1.set_title("Per-window discrimination vs lead time")
    fig.tight_layout(); fig.savefig(fig_dir / "discrimination_vs_lead.pdf"); plt.close(fig)


def _pick_examples(risk, n_per_class):
    k0 = risk[risk["k"] == 0].copy()
    out = {}
    out["TP"] = k0[(k0.y_true == 1) & (k0.ensemble_risk >= 0.5)].nlargest(n_per_class, "ensemble_risk")
    out["FN"] = k0[(k0.y_true == 1) & (k0.ensemble_risk < 0.5)].nsmallest(n_per_class, "ensemble_risk")
    out["FP"] = k0[(k0.y_true == 0) & (k0.ensemble_risk >= 0.5)].nlargest(n_per_class, "ensemble_risk")
    out["TN"] = k0[(k0.y_true == 0) & (k0.ensemble_risk < 0.5)].nsmallest(n_per_class, "ensemble_risk")
    return out


def fig_examples(risk, shap_dir, fig_dir=FIG_DIR, n_per_class=1, top_k=6):
    fig_dir = _ensure(fig_dir)
    names = json.loads((shap_dir / "feature_names.json").read_text())
    head_key = {"EEG": "eeg", "CT": "ct", "MRI": "mri"}
    for cls, sel in _pick_examples(risk, n_per_class).items():
        for uid in sel["uid"]:
            rg = risk[risk["uid"] == uid].sort_values("k")
            z = np.load(shap_dir / f"{uid}.npz")
            order = np.argsort(z["k"])
            ks = z["k"][order]
            # choose the head whose component prob is largest at k=0 for this patient
            comp_at0 = rg.iloc[(rg["k"] == 0).argmax()][["EEG_p", "CT_p", "MRI_p"]].to_numpy()
            head = COMPONENTS[int(np.argmax(comp_at0))]
            S = z[head_key[head]][order][:, :811]                 # drop bias col
            top = np.argsort(np.abs(S).max(axis=0))[::-1][:top_k]

            fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1.4]})
            a1.plot(rg["k"], rg["ensemble_risk"], color="black", lw=2)
            a1.axhline(0.5, color="gray", ls="--", lw=0.8)
            a1.set_ylabel("ensemble risk"); a1.set_title(f"{cls}  uid={uid}  (SHAP head: {head})")
            for j in top:
                a2.plot(ks, S[:, j], label=names[j])
            a2.axhline(0, color="gray", lw=0.6)
            a2.set_xlabel("hours before anchor (k; 0 = tested window)")
            a2.set_ylabel(f"{head} head SHAP (logit)")
            a2.legend(fontsize=7, ncol=2)
            a1.invert_xaxis()                                     # admission -> anchor left-to-right
            fig.tight_layout(); fig.savefig(fig_dir / f"example_{cls}_{uid}.pdf"); plt.close(fig)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/test_figures_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/make_figures.py" "Predictive Modeling/Risk Trajectories/tests/test_figures_smoke.py"
git commit -m "feat: trajectory figures (aggregate risk, discrimination, examples)"
```

---

### Task 12: Full run + README

**Files:**
- Create: `Predictive Modeling/Risk Trajectories/README.md`
- Produces: `artifacts/risk_trajectories.parquet`, `artifacts/shap/<uid>.npz` (×3895), `artifacts/window_metrics.csv`, `figures/*.pdf`

- [ ] **Step 1: Full test suite**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY -m pytest tests/ -v`
Expected: all PASS (including the k=0 gate).

- [ ] **Step 2: Smoke-run the driver on 25 patients first**

Run: `cd "Predictive Modeling/Risk Trajectories" && $PY compute_trajectories.py --limit 25 --chunk-size 25`
Expected: prints a chunk progress line and `done -> .../artifacts/risk_trajectories.parquet`. Then verify k=0 metrics on this subset:
Run: `$PY window_metrics.py`
(Subset AUPRC will be noisy on 25 patients; this just confirms the wiring end-to-end.)

- [ ] **Step 3: Full run (all 3,895 patients)**

Run (consider `nohup`/`caffeinate` for a long job):
```bash
cd "Predictive Modeling/Risk Trajectories"
$PY compute_trajectories.py --chunk-size 50
```
Expected: progress lines per chunk; on completion `artifacts/shap/` holds ~3,895 `.npz` files and `risk_trajectories.parquet` has the full set. This is the long step (spec §12: minutes–low hours; watch memory stays flat across chunks).

- [ ] **Step 4: Compute metrics and figures**

```bash
$PY window_metrics.py
$PY -c "import pandas as pd, make_figures as MF; r=pd.read_parquet('artifacts/risk_trajectories.parquet'); from pathlib import Path; d=Path('figures'); MF.fig_aggregate_risk(r,d); MF.fig_window_metrics(r,d); MF.fig_examples(r, Path('artifacts/shap'), d, n_per_class=2)"
```
Expected: `window_metrics.py` prints the **headline average AUPRC** and a `k=0 AUPRC≈0.7913 / AUROC≈0.9229` line; `figures/` contains the aggregate, discrimination, and example PDFs.

- [ ] **Step 5: Confirm the k=0 anchor matches the published metric**

Run:
```bash
$PY -c "import pandas as pd, window_metrics as WM; per,ov=WM.compute(pd.read_parquet('artifacts/risk_trajectories.parquet')); row=per[per.k==0].iloc[0]; print('k0 AUPRC',round(row.AUPRC,4),'AUROC',round(row.AUROC,4)); assert abs(row.AUPRC-0.7913)<0.01 and abs(row.AUROC-0.9229)<0.01, 'k=0 does not match published ensemble metrics'"
```
Expected: prints `k0 AUPRC 0.7913 AUROC 0.9229` (within 0.01) and no assertion error. If it fails, the full run drifted from the gate — investigate before trusting the trajectories.

- [ ] **Step 6: Write `README.md`**

Document: what was computed; the window config (window 48h, censor 12h, **hourly** stride); the SAITS checkpoint id; the artifact schema (`risk_trajectories.parquet` columns; `shap/<uid>.npz` arrays `eeg/ct/mri` `(n_windows,812)`, `meta` `(n_windows,4)`, `k`; `feature_names.json`); how to load one patient; the headline average AUPRC and the k=0 = published-metric check; and the **length-selection bias** caveat (n shrinks with k). Include the reproduce commands and the env path.

- [ ] **Step 7: Commit**

```bash
git add "Predictive Modeling/Risk Trajectories/README.md"
git commit -m "docs: Risk Trajectories README + full-run results"
```
(The `artifacts/` and `figures/` outputs are large — decide per repo convention whether to commit, `.gitignore`, or store outside git. Default: add `artifacts/shap/` and `*.parquet` to a local `.gitignore` and keep figures + `window_metrics.csv`.)

---

## Self-Review (completed by plan author)

**Spec coverage:** §1 goal → Tasks 9–11. §2 background → Tasks 1,3,4,7. §3 windowing (hourly) → Task 2. §4 faithful features → Tasks 3–5,8. §5 scoring + biomarker SHAP (all heads, exact TreeSHAP) → Task 6, persisted in Task 9. §6 validation gate → Task 8. §7 bounded memory/chunked streaming → Tasks 7,9. §8 artifacts (risk parquet, per-patient npz, window_metrics, README) → Tasks 9,10,12. §9 charts → Task 11. §10 code structure → all tasks (one focused module each). §11 non-goals respected (no retraining/refit). §12 risks → Task 8 fallback steps + Task 12 step 5 anchor check.

**Placeholder scan:** every code/test step contains complete runnable code; no TBD/"handle edge cases"/"similar to". Diagnostic fallback in Task 8 is explicit steps, not a placeholder.

**Type/name consistency:** `Bundle`/`load_models`/`predict_components`/`predict_ensemble`/`head_shap`/`meta_shap` (model.py) used identically in Tasks 6,8,9. `Engine.features_for_windows`/`windows_for_patient` used in Tasks 8,9. `Window(k,w_start,w_end)` consistent across Tasks 2,8,9. `data_io.test_uids/build_test_raw/iter_patient_frames` consistent across Tasks 7,8,9,11. SHAP npz keys `eeg/ct/mri/meta/k` consistent across Tasks 9,11. `window_metrics.compute` signature consistent across Tasks 10,11.
