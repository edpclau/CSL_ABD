"""Cache faithful Permutation-based biomarker -> outcome SHAP for ALL test patients
(at the k=0 / tested window = X_test_control). Values are in meta-LOGIT space:
    outcome_logit(x) = bias + sum_j phi_j(x)      (additive; bias = E_background[logit]).

Parallelized across processes (catch22 isn't involved here, but the per-instance
PermutationExplainer is the cost). Parent builds one shared background, launches N
workers over round-robin uid shards, then merges to a single parquet.

Usage:
    python cache_outcome_shap.py [--nshards 8]
    python cache_outcome_shap.py --worker I --nshards N   # internal
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import X_TEST_CONTROL, CONTROL, FEATURE_LIST_811, OUT_DIR

BG_PATH = OUT_DIR / "outcome_bg.npy"
SHARD_DIR = OUT_DIR / "outcome_shap_shards"
OUT_PARQUET = OUT_DIR / "outcome_shap_permutation.parquet"
META_PATH = OUT_DIR / "outcome_shap_meta.json"
MAX_EVALS = 10000
N_BG = 64
BG_SEED = 7


def _test_uids():
    return list(map(str, pd.read_csv(X_TEST_CONTROL, index_col=0).index))


def build_background():
    rng = np.random.default_rng(BG_SEED)
    Xtr = pd.read_csv(CONTROL / "X_train_control.csv", index_col=0)[FEATURE_LIST_811]
    bg = Xtr.iloc[rng.choice(len(Xtr), N_BG, replace=False)].to_numpy(np.float32)
    BG_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(BG_PATH, bg)
    return bg


def worker(shard, nshards):
    import warnings; warnings.filterwarnings("ignore")
    from outcome_shap import OutcomeAttributor
    bg = np.load(BG_PATH)
    Xte = pd.read_csv(X_TEST_CONTROL, index_col=0)[FEATURE_LIST_811]
    my = list(Xte.index)[shard::nshards]
    att = OutcomeAttributor(bg)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[worker {shard}] {len(my)} patients", flush=True)
    phis, fxs, uids = [], [], []
    for a in range(0, len(my), 50):
        chunk = my[a:a + 50]
        Xc = Xte.loc[chunk].to_numpy(np.float32)
        phis.append(att.attribute(Xc, "permutation", max_evals=MAX_EVALS).astype(np.float32))
        fxs.append(att._f(Xc).astype(np.float32))
        uids.extend(map(str, chunk))
        print(f"[worker {shard}] {min(a+50, len(my))}/{len(my)}", flush=True)
    np.savez_compressed(SHARD_DIR / f"shard{shard}.npz",
                        uids=np.array(uids), phi=np.concatenate(phis), fx=np.concatenate(fxs))
    print(f"[worker {shard}] done", flush=True)


def parent(nshards):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[parent] building {N_BG}-row training background ...", flush=True)
    bg = build_background()
    from outcome_shap import OutcomeAttributor
    base = float(OutcomeAttributor(bg)._f(bg).mean())
    print(f"[parent] base (E_bg[logit]) = {base:.4f}; launching {nshards} workers", flush=True)

    procs = []
    for i in range(nshards):
        logf = open(OUT_DIR / f"outcome_shard{i}.log", "w")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--nshards", str(nshards)]
        procs.append((i, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf))
    failed = []
    for i, p, logf in procs:
        rc = p.wait(); logf.close()
        print(f"[parent] worker {i} exited rc={rc}", flush=True)
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"workers failed: {failed} (see artifacts/outcome_shard*.log)")

    parts = [np.load(f, allow_pickle=True) for f in sorted(SHARD_DIR.glob("shard*.npz"))]
    uids = np.concatenate([p["uids"] for p in parts])
    phi = np.concatenate([p["phi"] for p in parts])
    fx = np.concatenate([p["fx"] for p in parts])
    df = pd.DataFrame(phi, columns=FEATURE_LIST_811, index=pd.Index(uids, name="uid"))
    df["bias"] = base
    df["outcome_logit"] = fx
    df = df.sort_index()
    tmp = OUT_DIR / "outcome_shap_permutation.tmp.parquet"
    df.to_parquet(tmp); tmp.replace(OUT_PARQUET)
    for f in SHARD_DIR.glob("shard*.npz"):
        f.unlink()
    recon = float(np.max(np.abs((df[FEATURE_LIST_811].sum(1) + base) - df["outcome_logit"])))
    META_PATH.write_text(json.dumps({
        "method": "permutation", "space": "meta_logit", "max_evals": MAX_EVALS,
        "n_background": N_BG, "background": "seeded sample of X_train_control",
        "bg_seed": BG_SEED, "base_value": base, "n_patients": int(len(df)),
        "n_features": len(FEATURE_LIST_811), "additivity_max_resid": recon,
    }, indent=2))
    print(f"[parent] cached {len(df)} patients x {len(FEATURE_LIST_811)} -> {OUT_PARQUET}  "
          f"(additivity resid {recon:.1e})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--worker", type=int, default=None)
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.nshards)
    else:
        t0 = time.time()
        parent(args.nshards)
        print(f"[parent] total wall {(time.time()-t0)/60:.1f} min", flush=True)
