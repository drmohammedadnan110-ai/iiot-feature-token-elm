import pytest

from iiot_ids.data import prepare
from iiot_ids.demo import synthetic_data


@pytest.fixture
def prepared(tmp_path):
    csv, schema = synthetic_data(tmp_path / "raw", ["Normal", "Attack_A", "Attack_B"], minority_rows=30)
    return prepare([csv], schema, tmp_path / "prepared")


@pytest.fixture
def tiny_config(prepared):
    return {
        "data_dir": str(prepared),
        "seed": 7,
        "threads": 2,
        "device": "cpu",
        "model": {"width": 8, "layers": 1, "heads": 2, "expansion": 2, "dropout": 0.0},
        "training": {"epochs": 2, "patience": 2, "batch_size": 32, "lr": 0.005},
        "elm": {"hidden": 12, "ridge": 0.1, "activation": "tanh", "initialization": "orthogonal"},
    }
