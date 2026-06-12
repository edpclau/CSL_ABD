# tests/test_config.py
import config

def test_biomarker_cols_match_feature_list_prefixes():
    prefixes, seen = [], set()
    for f in config.FEATURE_LIST_811:
        p = f.rsplit("__", 1)[0]
        if p not in seen:
            seen.add(p); prefixes.append(p)
    assert config.BIOMARKER_COLS == prefixes
    assert len(config.BIOMARKER_COLS) == 45

def test_feature_list_has_811():
    assert len(config.FEATURE_LIST_811) == 811

def test_key_paths_exist():
    assert config.NMB.exists()
    assert config.SAITS_CKPT.exists()
    assert config.FEATURE_LIST.exists()
    assert config.X_TEST_CONTROL.exists()
    assert config.OUTCOME_COMPONENTS_TEST.exists()
    for h in ("EEG", "CT", "MRI"):
        assert (config.ENS_TUNED / f"head_{h}_tuned.json").exists()
    assert (config.ENS_TUNED / "meta_xgb_tuned.json").exists()

def test_catch22_has_22_features():
    assert len(config.CATCH22_FEATURES) == 22
