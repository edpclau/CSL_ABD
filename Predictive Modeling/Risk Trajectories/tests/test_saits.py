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
