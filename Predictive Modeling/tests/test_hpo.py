import pandas as pd
from sklearn.datasets import make_classification

import hpo
from hpo import config_label


def test_config_label_full():
    assert config_label(False, False) == "full"


def test_config_label_no_drugs():
    assert config_label(True, False) == "no_drugs"


def test_config_label_vitals_labs():
    assert config_label(True, True) == "vitals_labs"


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
