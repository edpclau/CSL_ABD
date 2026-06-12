# tests/test_catch22.py
import numpy as np
import featurize as F
from config import BIOMARKER_COLS, FEATURE_LIST_811, N_STEPS

def test_catch22_outputs_811_named_columns_in_order():
    feat = F.Catch22Featurizer()
    rng = np.random.default_rng(0)                     # deterministic
    # several windows of imputed (no-NaN) data, shape (n, N_STEPS, 45)
    arr = rng.random((6, N_STEPS, len(BIOMARKER_COLS))).astype(np.float32)
    out = feat.transform_batch(arr)
    assert list(out.columns) == FEATURE_LIST_811       # exact 811, exact order
    assert out.shape == (6, 811)
    # every requested feature name matched a catch22 output column:
    # an unmatched name would make the WHOLE column NaN (fully-NaN). Sporadic
    # per-cell NaN from degenerate features is expected and tolerated.
    assert out.isna().all(axis=0).sum() == 0           # no fully-NaN columns
