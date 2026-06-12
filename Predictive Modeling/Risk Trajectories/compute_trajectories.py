# compute_trajectories.py — streaming, chunked driver. Writes risk parquet + per-patient SHAP.
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import data_io
from config import (OUT_DIR, FEATURE_LIST_811, OUTCOME_COMPONENTS_TEST)
from pipeline import Engine
from model import load_models, predict_components, predict_ensemble, head_shap, meta_shap

_RISK_COLS = ["uid", "k", "n_observed", "EEG_p", "CT_p", "MRI_p", "ensemble_risk", "y_true"]


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run(uids=None, chunk_size=50, out_dir=None, log_every=1, raw=None,
        risk_filename="risk_trajectories.parquet"):
    log_every = max(1, log_every)
    out_dir = OUT_DIR if out_dir is None else out_dir
    shap_dir = out_dir / "shap"
    risk_parquet = out_dir / risk_filename
    shap_dir.mkdir(parents=True, exist_ok=True)
    (shap_dir / "feature_names.json").write_text(json.dumps(FEATURE_LIST_811))

    all_uids = data_io.test_uids() if uids is None else list(map(str, uids))
    # `raw` (pre-built compact test table) lets parallel workers share one build
    # and avoid concurrently re-streaming/overwriting test_raw.parquet.
    raw = data_io.build_test_raw(all_uids) if raw is None else raw
    ytrue = pd.read_csv(OUTCOME_COMPONENTS_TEST, index_col=0)["Outcome"].astype(int)
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
