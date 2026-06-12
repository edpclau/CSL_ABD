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
