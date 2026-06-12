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
