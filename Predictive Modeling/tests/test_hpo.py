from hpo import config_label
import pandas as pd
import hpo


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
