from hpo import config_label


def test_config_label_full():
    assert config_label(False, False) == "full"


def test_config_label_no_drugs():
    assert config_label(True, False) == "no_drugs"


def test_config_label_vitals_labs():
    assert config_label(True, True) == "vitals_labs"
