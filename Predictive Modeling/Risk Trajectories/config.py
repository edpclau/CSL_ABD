# config.py — paths and constants for the moving-window trajectory pipeline.
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # .../Predictive Modeling/Risk Trajectories
PM = SCRIPT_DIR.parent                                 # .../Predictive Modeling
AIM1 = SCRIPT_DIR.parents[2]                            # .../Aim 1 — shared data root, one level ABOVE the repo root (Aim 1.1)

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
DROP_COLS = frozenset(CONFOUNDING_COLS + BIAS_COLS + OUTCOME_COLS + ["arrive_yr", "uid", "timestamp"])

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
del _seen, _feat, _p
