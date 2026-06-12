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
