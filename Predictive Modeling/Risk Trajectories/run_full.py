# run_full.py — parallel orchestrator for the full trajectory run.
#
# catch22 (sktime) is the bottleneck (~94% of time) and is single-threaded, so we
# parallelize the EXACT same pipeline across processes: the parent builds the
# compact test table once, then launches N worker processes over round-robin
# shards of the test uids. Each worker runs the identical Engine/Catch22 path
# (so values are bit-identical to the single-process driver) and writes:
#   - per-patient SHAP to the SHARED artifacts/shap/<uid>.npz  (distinct uids, no conflict)
#   - its risk rows to artifacts/risk_shard<i>.parquet
# The parent then merges the shard parquets into artifacts/risk_trajectories.parquet.
#
# Usage:
#   python run_full.py [--nshards 6] [--limit N]      # parent (build + launch + merge)
#   python run_full.py --worker I --nshards N [--limit N]   # worker (internal)
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import data_io
from config import OUT_DIR, FEATURE_LIST_811


def _existing_uids():
    """uids already written to any completed/partial risk_shard*.parquet."""
    done = set()
    for f in OUT_DIR.glob("risk_shard*.parquet"):
        done |= set(pd.read_parquet(f, columns=["uid"])["uid"].astype(str).unique())
    return done


def _all_uids(limit):
    uids = data_io.test_uids()
    return uids[:limit] if limit is not None else uids


def _work_uids(limit, shard, nshards, resume):
    pool = [u for u in _all_uids(limit) if u not in _existing_uids()] if resume else _all_uids(limit)
    return pool[shard::nshards]   # round-robin balances long/short stays across workers


def worker(shard, nshards, limit, resume):
    import compute_trajectories as C
    raw = data_io.load_test_raw()                       # parent already built it
    uids = _work_uids(limit, shard, nshards, resume)
    tag = "resume" if resume else "shard"
    print(f"[worker {shard}] {len(uids)} patients ({tag})", flush=True)
    C.run(uids=uids, raw=raw, out_dir=OUT_DIR, chunk_size=25,
          risk_filename=f"risk_{tag}{shard}.parquet")
    print(f"[worker {shard}] done", flush=True)


def parent(nshards, limit, resume):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "shap").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "shap" / "feature_names.json").write_text(json.dumps(FEATURE_LIST_811))

    all_uids = _all_uids(limit)
    if resume:
        missing = [u for u in all_uids if u not in _existing_uids()]
        print(f"[parent] resume: {len(missing)} of {len(all_uids)} patients missing; "
              f"reusing existing shards; launching {nshards} workers", flush=True)
        if not missing:
            print("[parent] nothing missing; merging existing shards", flush=True)
    else:
        print(f"[parent] building compact test table for {len(all_uids)} patients ...", flush=True)
        t0 = time.time()
        data_io.build_test_raw(all_uids)                # writes artifacts/test_raw.parquet once
        print(f"[parent] test table built in {time.time()-t0:.0f}s; launching {nshards} workers", flush=True)

    procs = []
    for i in range(nshards):
        logf = open(OUT_DIR / f"{'resume' if resume else 'shard'}{i}.log", "w")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", str(i), "--nshards", str(nshards)]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if resume:
            cmd += ["--resume"]
        procs.append((i, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf))

    failed = []
    for i, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"[parent] worker {i} exited rc={rc}", flush=True)
        if rc != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"workers failed: {failed} (see artifacts/shard*.log)")

    shard_files = sorted(OUT_DIR.glob("risk_shard*.parquet")) + sorted(OUT_DIR.glob("risk_resume*.parquet"))
    merged = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    merged = merged.drop_duplicates(subset=["uid", "k"]).sort_values(["uid", "k"]).reset_index(drop=True)
    out = OUT_DIR / "risk_trajectories.parquet"
    merged.to_parquet(out)
    for f in shard_files:
        f.unlink()
    print(f"[parent] merged {len(shard_files)} shard files -> {out}  "
          f"({len(merged)} window-rows, {merged['uid'].nunique()} patients)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="only process uids missing from existing risk_shard*.parquet")
    args = ap.parse_args()
    if args.worker is not None:
        worker(args.worker, args.nshards, args.limit, args.resume)
    else:
        t0 = time.time()
        parent(args.nshards, args.limit, args.resume)
        print(f"[parent] total wall time {(time.time()-t0)/60:.1f} min", flush=True)
