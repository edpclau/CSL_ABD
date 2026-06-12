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
