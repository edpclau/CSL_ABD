# tests/test_validate_k0.py
import validate_k0

def test_k0_reproduces_saved_features_and_risk():
    feat_max, feat_corr, risk_max = validate_k0.run(n_patients=40)
    assert feat_corr > 0.999
    assert risk_max < 5e-3
