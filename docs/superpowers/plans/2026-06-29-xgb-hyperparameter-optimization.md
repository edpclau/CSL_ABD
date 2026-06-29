# XGBoost Hyperparameter Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Optuna-based XGBoost hyperparameter optimization, tuned once per (DAG, config) and cached to JSON, reused across both modeling notebooks without test-set leakage.

**Architecture:** A standalone `hpo.py` module in `Predictive Modeling/` exposes three functions (`config_label`, `tune_xgb`, `get_xgb_params`). Both notebooks import it and replace their fixed `XGBClassifier(...)` instantiation with cache-aware tuned params. Tuning uses Optuna (TPE + MedianPruner) over a `StratifiedKFold` loop scoring `average_precision` on training data only.

**Tech Stack:** Python 3.12, Optuna, XGBoost ≥3, scikit-learn, pandas, pytest, pixi.

## Global Constraints

- All paths are relative to `Aim 1.1 Causal Discovery/Predictive Modeling/` unless noted.
- Run all Python/tests through pixi: `pixi run python ...`, `pixi run pytest ...`.
- Tuning objective is `average_precision`; budget defaults `n_trials=50`, `cv=5`, `seed=42`.
- XGB base params are exactly `dict(objective='binary:logistic', random_state=42, eval_metric='aucpr')`.
- Tuning and CV use TRAINING data only. `X_test`/`X_test_imp` must never be passed to `tune_xgb`/`get_xgb_params`.
- `scale_pos_weight` is NOT in the search space (preserve probability calibration).
- LGBN/pgmpy code paths are untouched.
- Cache key format: `f"{dag_name}||{cfg_label}"`.

---

### Task 1: Add optuna + pytest dependencies

**Files:**
- Modify: `pixi.toml` (`[dependencies]` table)

**Interfaces:**
- Produces: `optuna` and `pytest` importable via `pixi run python`.

- [ ] **Step 1: Add the dependencies to `pixi.toml`**

In `pixi.toml`, under `[dependencies]`, add these two lines after `ipykernel = "*"`:

```toml
optuna = "*"
pytest = "*"
```

- [ ] **Step 2: Install and verify both import**

Run: `pixi run python -c "import optuna, pytest; print('optuna', optuna.__version__); print('pytest', pytest.__version__)"`
Expected: prints versions for both, no `ModuleNotFoundError`.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build: add optuna and pytest to pixi env"
```

---

### Task 2: `config_label` in hpo.py

**Files:**
- Create: `Predictive Modeling/hpo.py`
- Test: `Predictive Modeling/tests/test_hpo.py`

**Interfaces:**
- Produces: `config_label(remove_drugs: bool, remove_interventions: bool) -> str` returning one of `"full"`, `"no_drugs"`, `"vitals_labs"`.

- [ ] **Step 1: Write the failing test**

Create `Predictive Modeling/tests/test_hpo.py`:

```python
from hpo import config_label


def test_config_label_full():
    assert config_label(False, False) == "full"


def test_config_label_no_drugs():
    assert config_label(True, False) == "no_drugs"


def test_config_label_vitals_labs():
    assert config_label(True, True) == "vitals_labs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hpo'`.

- [ ] **Step 3: Write minimal implementation**

Create `Predictive Modeling/hpo.py`:

```python
"""XGBoost hyperparameter optimization with a per-(DAG, config) JSON cache.

Tuning uses Optuna over a StratifiedKFold loop scoring average_precision on
TRAINING data only. See docs/superpowers/specs/2026-06-29-xgb-hyperparameter-optimization-design.md
"""

FIXED_DEFAULTS = dict(objective="binary:logistic", random_state=42, eval_metric="aucpr")


def config_label(remove_drugs, remove_interventions):
    """Canonical config string shared as a cache key across both notebooks."""
    if remove_drugs and remove_interventions:
        return "vitals_labs"
    if remove_drugs:
        return "no_drugs"
    return "full"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/hpo.py" "Predictive Modeling/tests/test_hpo.py"
git commit -m "feat: hpo.config_label canonical config keys"
```

---

### Task 3: Cache helpers + `get_xgb_params`

**Files:**
- Modify: `Predictive Modeling/hpo.py`
- Test: `Predictive Modeling/tests/test_hpo.py`

**Interfaces:**
- Consumes: `config_label` (Task 2).
- Produces:
  - `get_xgb_params(dag_name, cfg_label, X, y, cache_path, tune=True, force_retune=False, n_trials=50, cv=5, seed=42) -> dict`
  - On `tune=False` returns `{}` (no cache I/O). On cache hit returns `cache[key]["params"]`. On miss/`force_retune` calls `tune_xgb`, writes the entry, returns params.
  - Internal `_load_cache(path) -> dict`, `_save_cache(path, cache) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `Predictive Modeling/tests/test_hpo.py`:

```python
import pandas as pd
import hpo


def _toy_xy(n=40):
    X = pd.DataFrame({"a": list(range(n)), "b": list(range(n, 2 * n))})
    y = pd.Series([0, 1] * (n // 2))
    return X, y


def test_tune_false_returns_empty_and_writes_nothing(tmp_path):
    cache = tmp_path / "cache.json"
    X, y = _toy_xy()
    out = hpo.get_xgb_params("DAG", "full", X, y, cache_path=str(cache), tune=False)
    assert out == {}
    assert not cache.exists()


def test_cache_miss_then_hit(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_tune(X, y, n_trials=50, cv=5, seed=42):
        calls["n"] += 1
        return {"max_depth": 4, "n_estimators": 123}

    monkeypatch.setattr(hpo, "tune_xgb", fake_tune)
    cache = tmp_path / "cache.json"
    X, y = _toy_xy()

    first = hpo.get_xgb_params("DAG", "full", X, y, cache_path=str(cache))
    assert first == {"max_depth": 4, "n_estimators": 123}
    assert cache.exists()
    assert calls["n"] == 1

    second = hpo.get_xgb_params("DAG", "full", X, y, cache_path=str(cache))
    assert second == {"max_depth": 4, "n_estimators": 123}
    assert calls["n"] == 1  # served from cache, tuner not called again


def test_force_retune_calls_tuner_again(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_tune(X, y, n_trials=50, cv=5, seed=42):
        calls["n"] += 1
        return {"max_depth": calls["n"]}

    monkeypatch.setattr(hpo, "tune_xgb", fake_tune)
    cache = tmp_path / "cache.json"
    X, y = _toy_xy()

    hpo.get_xgb_params("DAG", "full", X, y, cache_path=str(cache))
    out = hpo.get_xgb_params("DAG", "full", X, y, cache_path=str(cache), force_retune=True)
    assert calls["n"] == 2
    assert out == {"max_depth": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -k "cache or tune_false or force" -v`
Expected: FAIL — `AttributeError: module 'hpo' has no attribute 'get_xgb_params'`.

- [ ] **Step 3: Write minimal implementation**

Append to `Predictive Modeling/hpo.py`:

```python
import json
import os
from datetime import datetime


def _load_cache(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path, cache):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def get_xgb_params(dag_name, cfg_label, X, y, cache_path,
                   tune=True, force_retune=False, n_trials=50, cv=5, seed=42):
    """Return tuned XGB params for (dag_name, cfg_label), using a JSON cache.

    tune=False -> {} (reproduces fixed defaults, no cache I/O).
    Cache hit  -> stored params. Miss/force_retune -> tune, cache, return.
    """
    if not tune:
        return {}
    key = f"{dag_name}||{cfg_label}"
    cache = _load_cache(cache_path)
    if key in cache and not force_retune:
        return cache[key]["params"]
    params = tune_xgb(X, y, n_trials=n_trials, cv=cv, seed=seed)
    cache[key] = {
        "params": params,
        "n_features": int(X.shape[1]),
        "n_trials": n_trials,
        "cv": cv,
        "objective": "average_precision",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _save_cache(cache_path, cache)
    return params
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -k "cache or tune_false or force" -v`
Expected: 3 passed. (`tune_xgb` is referenced but only via monkeypatch here; the real one lands in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/hpo.py" "Predictive Modeling/tests/test_hpo.py"
git commit -m "feat: hpo.get_xgb_params cache-aware param lookup"
```

---

### Task 4: `tune_xgb` Optuna implementation

**Files:**
- Modify: `Predictive Modeling/hpo.py`
- Test: `Predictive Modeling/tests/test_hpo.py`

**Interfaces:**
- Consumes: `FIXED_DEFAULTS` (Task 2).
- Produces: `tune_xgb(X, y, n_trials=50, cv=5, seed=42) -> dict` returning `study.best_params` (a subset of the keys: `n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha, gamma`).

- [ ] **Step 1: Write the failing test**

Append to `Predictive Modeling/tests/test_hpo.py`:

```python
from sklearn.datasets import make_classification


def test_tune_xgb_smoke_returns_param_dict():
    Xa, ya = make_classification(
        n_samples=120, n_features=8, weights=[0.8, 0.2], random_state=0
    )
    X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(ya)
    params = hpo.tune_xgb(X, y, n_trials=4, cv=3, seed=42)
    assert isinstance(params, dict)
    assert params  # non-empty
    allowed = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha", "gamma",
    }
    assert set(params).issubset(allowed)


def test_tune_xgb_is_deterministic():
    Xa, ya = make_classification(
        n_samples=120, n_features=8, weights=[0.8, 0.2], random_state=0
    )
    X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(8)])
    y = pd.Series(ya)
    p1 = hpo.tune_xgb(X, y, n_trials=4, cv=3, seed=42)
    p2 = hpo.tune_xgb(X, y, n_trials=4, cv=3, seed=42)
    assert p1 == p2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -k "tune_xgb" -v`
Expected: FAIL — `AttributeError: module 'hpo' has no attribute 'tune_xgb'`.

- [ ] **Step 3: Write minimal implementation**

Append to `Predictive Modeling/hpo.py`:

```python
import numpy as np
import optuna
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def tune_xgb(X, y, n_trials=50, cv=5, seed=42):
    """Optuna TPE search (MedianPruner) maximizing CV average_precision.

    Uses TRAINING data only. Returns study.best_params.
    """
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    yv = np.asarray(y)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        for fold, (tr, va) in enumerate(skf.split(Xv, yv)):
            model = xgboost.XGBClassifier(**FIXED_DEFAULTS, **params)
            model.fit(Xv[tr], yv[tr])
            prob = model.predict_proba(Xv[va])[:, 1]
            scores.append(average_precision_score(yv[va], prob))
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `cd "Predictive Modeling" && pixi run pytest tests/test_hpo.py -v`
Expected: all tests pass (config_label ×3, cache ×3, tune_xgb ×2).

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/hpo.py" "Predictive Modeling/tests/test_hpo.py"
git commit -m "feat: hpo.tune_xgb Optuna TPE + MedianPruner search"
```

---

### Task 5: Integrate HPO into DAG Parameterization.ipynb

**Files:**
- Modify: `Predictive Modeling/DAG Parameterization.ipynb` (imports cell 1; train cell 8)

**Interfaces:**
- Consumes: `config_label`, `get_xgb_params` (Tasks 2–4).

- [ ] **Step 1: Add config constants + import to the imports cell**

Run this from `Predictive Modeling/`:

```bash
pixi run python - <<'PY'
import json
f = "DAG Parameterization.ipynb"
nb = json.load(open(f))
setup = (
    "\n# --- Hyperparameter optimization (see hpo.py) ---\n"
    "from hpo import config_label, get_xgb_params\n"
    "CACHE_PATH = 'xgb_best_params.json'  # per-(DAG, config) Optuna cache\n"
    "TUNE = True            # False reproduces fixed XGB defaults\n"
    "FORCE_RETUNE = False   # True ignores cache and re-tunes\n"
    "N_TRIALS = 50\n"
)
cell = nb["cells"][1]
src = "".join(cell["source"])
assert "import xgboost" in src, "cell 1 is not the imports cell"
if "from hpo import" not in src:
    cell["source"] = src.splitlines(keepends=True) + [setup]
json.dump(nb, open(f, "w"), indent=1)
print("imports cell updated")
PY
```

- [ ] **Step 2: Replace the fixed XGB instantiation in train cell 8**

```bash
pixi run python - <<'PY'
import json
f = "DAG Parameterization.ipynb"
nb = json.load(open(f))
OLD = "models['XGB'] = xgboost.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='aucpr')"
NEW = (
    "_best = get_xgb_params(dag_name, config_label(remove_drugs, remove_interventions),\n"
    "                                       X_train.filter(xgb_features), y_train.Outcome,\n"
    "                                       cache_path=CACHE_PATH, tune=TUNE, force_retune=FORCE_RETUNE, n_trials=N_TRIALS)\n"
    "                models['XGB'] = xgboost.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='aucpr', **_best)"
)
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if OLD in s:
        c["source"] = s.replace(OLD, NEW).splitlines(keepends=True)
        hits += 1
json.dump(nb, open(f, "w"), indent=1)
print("XGB line replaced in", hits, "cell(s)")
PY
```

Expected: `XGB line replaced in 1 cell(s)`.

- [ ] **Step 3: Verify the integration path runs (fast, isolated)**

This replicates the exact notebook call on one tiny feature set, writing to a temp cache — proving the wiring without the multi-hour notebook run:

```bash
pixi run python - <<'PY'
import pandas as pd
from sklearn.datasets import make_classification
from hpo import config_label, get_xgb_params
import xgboost, os, json
Xa, ya = make_classification(n_samples=150, n_features=6, weights=[0.8, 0.2], random_state=1)
X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(6)]); y = pd.Series(ya)
cache = "/tmp/_hpo_wire.json"
if os.path.exists(cache): os.remove(cache)
best = get_xgb_params("Simplified PCMB", config_label(False, False), X, y,
                      cache_path=cache, tune=True, force_retune=False, n_trials=3)
m = xgboost.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='aucpr', **best)
m.fit(X, y)
assert "Simplified PCMB||full" in json.load(open(cache))
print("wiring OK; best params:", best)
PY
```

Expected: prints `wiring OK; best params: {...}` with no assertion error.

- [ ] **Step 4: Verify notebook still parses as valid JSON**

Run: `pixi run python -c "import json; json.load(open('DAG Parameterization.ipynb')); print('valid notebook')"`
Expected: `valid notebook`.

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/DAG Parameterization.ipynb"
git commit -m "feat: wire Optuna HPO into DAG Parameterization notebook"
```

---

### Task 6: Integrate HPO into Year Sensitivity notebook

**Files:**
- Modify: `Predictive Modeling/DAG Parameterization - Year Sensitivity.ipynb` (imports cell 1; train cell 11)

**Interfaces:**
- Consumes: `config_label`, `get_xgb_params`; reuses cache entries written by Task 5 (`full`, `vitals_labs`).

- [ ] **Step 1: Add config constants + import to the imports cell**

```bash
pixi run python - <<'PY'
import json
f = "DAG Parameterization - Year Sensitivity.ipynb"
nb = json.load(open(f))
setup = (
    "\n# --- Hyperparameter optimization (see hpo.py) ---\n"
    "from hpo import config_label, get_xgb_params\n"
    "CACHE_PATH = 'xgb_best_params.json'  # shared cache with DAG Parameterization\n"
    "TUNE = True\n"
    "FORCE_RETUNE = False\n"
    "N_TRIALS = 50\n"
)
cell = nb["cells"][1]
src = "".join(cell["source"])
assert "import xgboost" in src, "cell 1 is not the imports cell"
if "from hpo import" not in src:
    cell["source"] = src.splitlines(keepends=True) + [setup]
json.dump(nb, open(f, "w"), indent=1)
print("imports cell updated")
PY
```

- [ ] **Step 2: Replace the fixed XGB instantiation in train cell 11**

```bash
pixi run python - <<'PY'
import json
f = "DAG Parameterization - Year Sensitivity.ipynb"
nb = json.load(open(f))
OLD = "        xgb = xgboost.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='aucpr')\n"
NEW = (
    "        _best = get_xgb_params(dag_name, config_label(remove_drugs, remove_interventions),\n"
    "                              X_train.filter(features_in_dag), y_train.Outcome,\n"
    "                              cache_path=CACHE_PATH, tune=TUNE, force_retune=FORCE_RETUNE, n_trials=N_TRIALS)\n"
    "        xgb = xgboost.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='aucpr', **_best)\n"
)
hits = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    new = []
    changed = False
    for line in c["source"]:
        if line == OLD:
            new.append(NEW); changed = True; hits += 1
        else:
            new.append(line)
    if changed:
        c["source"] = new
json.dump(nb, open(f, "w"), indent=1)
print("XGB line replaced in", hits, "place(s)")
PY
```

Expected: `XGB line replaced in 1 place(s)`.

- [ ] **Step 3: Verify cache reuse (no re-tune on hit)**

```bash
pixi run python - <<'PY'
import pandas as pd, os, json
from sklearn.datasets import make_classification
import hpo
from hpo import config_label, get_xgb_params
Xa, ya = make_classification(n_samples=150, n_features=6, weights=[0.8, 0.2], random_state=1)
X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(6)]); y = pd.Series(ya)
cache = "/tmp/_hpo_reuse.json"
if os.path.exists(cache): os.remove(cache)
# Seed cache as DAG Parameterization would
get_xgb_params("Simplified PCMB", config_label(False, False), X, y, cache_path=cache, n_trials=3)
# Year Sensitivity hit must NOT call the tuner again
calls = {"n": 0}
orig = hpo.tune_xgb
def spy(*a, **k):
    calls["n"] += 1; return orig(*a, **k)
hpo.tune_xgb = spy
out = get_xgb_params("Simplified PCMB", config_label(False, False), X, y, cache_path=cache, n_trials=3)
assert calls["n"] == 0, "cache hit should not re-tune"
print("cache reuse OK; params:", out)
PY
```

Expected: prints `cache reuse OK; params: {...}` with no assertion error.

- [ ] **Step 4: Verify notebook still parses as valid JSON**

Run: `pixi run python -c "import json; json.load(open('DAG Parameterization - Year Sensitivity.ipynb')); print('valid notebook')"`
Expected: `valid notebook`.

- [ ] **Step 5: Commit**

```bash
git add "Predictive Modeling/DAG Parameterization - Year Sensitivity.ipynb"
git commit -m "feat: wire Optuna HPO into Year Sensitivity notebook"
```

---

### Task 7: Production tuned run (optional, expensive)

**Files:**
- Create (outputs): `Predictive Modeling/biomarker_counts_tuned/` (redirected like the earlier fixed run)

**Interfaces:**
- Consumes: integrated notebooks (Tasks 5–6).

> This task actually runs the full pipeline with tuning (≈11 DAGs × 3 configs × 50 trials × 5 folds, plus existing bootstrapping). Expect a long runtime. Skip if you only wanted the capability.

- [ ] **Step 1: Build redirected + tuned copies and run DAG Parameterization**

Reuse the redirect helper from the earlier fixed run: copy `DAG Parameterization.ipynb` to a `.tunedrun.ipynb`, replace `../Predictive Modeling/` with `biomarker_counts_tuned/` and `CACHE_PATH = 'xgb_best_params.json'` with `CACHE_PATH = 'biomarker_counts_tuned/xgb_best_params.json'`, create `biomarker_counts_tuned/{Calibration Curves,Feature Importance}/`, then:

```bash
pixi run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir "biomarker_counts_tuned" --output "DAG Parameterization.executed.ipynb" \
  "DAG Parameterization.tunedrun.ipynb"
```

Expected: exit 0; `biomarker_counts_tuned/xgb_best_params.json` populated; `Biomarker Selection.csv` etc. written.

- [ ] **Step 2: Run Year Sensitivity reusing the cache**

Same redirect for the Year Sensitivity notebook (pointing `CACHE_PATH` at `biomarker_counts_tuned/xgb_best_params.json`), then nbconvert-execute it. Confirm from its log that no tuning occurs (cache hits for `full` and `vitals_labs`).

- [ ] **Step 3: Commit outputs**

```bash
git add "Predictive Modeling/biomarker_counts_tuned"
git commit -m "results: tuned XGB run (Optuna HPO) for both notebooks"
```

---

## Self-Review

**Spec coverage:**
- `config_label` → Task 2. `tune_xgb` (Optuna/TPE/MedianPruner, AUPRC, StratifiedKFold, pruning) → Task 4. `get_xgb_params` (cache hit/miss/force/tune=False) → Task 3. Search space (no `scale_pos_weight`) → Task 4. Cache JSON format/key → Task 3. Integration both notebooks + no-leakage (train-only `X`) → Tasks 5–6. `optuna` dependency → Task 1. Cost/production run → Task 7. Reproducibility (seeds) → Task 4 + `test_tune_xgb_is_deterministic`. Testing section → Tasks 2–4. All covered.

**Placeholder scan:** No TBD/TODO; every code step contains full code; commands have expected output.

**Type consistency:** `get_xgb_params(dag_name, cfg_label, X, y, cache_path, tune, force_retune, n_trials, cv, seed)` and `tune_xgb(X, y, n_trials, cv, seed)` signatures are identical across definition (Tasks 3–4), tests, and notebook call sites (Tasks 5–6). Cache key `f"{dag_name}||{cfg_label}"` consistent. `FIXED_DEFAULTS` defined in Task 2, used in Tasks 3–4 and notebook base params.
