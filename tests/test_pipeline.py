import numpy as np
import pandas as pd
import pytest

from iiot_ids.data import read_csv
from iiot_ids.engine import Pipeline, fit
from iiot_ids.engine import test_bundle as evaluate_bundle
from iiot_ids.utils import read_json, write_json


def test_train_without_test_file_then_frozen_roundtrip(tiny_config, prepared, tmp_path):
    test_path = prepared / "test.csv"
    contents = test_path.read_bytes()
    test_path.unlink()
    root = tmp_path / "model"
    bundle = fit(tiny_config, root)
    assert "TEST not evaluated" in bundle["status"]
    test_path.write_bytes(contents)
    pipeline = Pipeline(root)
    assert all(not p.requires_grad for p in pipeline.encoder.parameters())
    with (
        np.load(root / "encoder_training.npz", allow_pickle=False) as training,
        np.load(root / "encoder.npz", allow_pickle=False) as frozen,
    ):
        for name in frozen.files:
            np.testing.assert_array_equal(frozen[name], training["encoder." + name])
    validation = read_csv(prepared / "val.csv")
    recomputed = pipeline.predict(validation)
    saved = pd.read_csv(root / "validation_predictions.csv")
    columns = [f"prob_{i}" for i in range(3)]
    np.testing.assert_allclose(recomputed[columns], saved[columns], atol=2e-6)
    result = evaluate_bundle(root, prepared, tmp_path / "test")
    assert result["n"] == len(read_csv(test_path))
    test_path.write_bytes(contents + b"\n")
    with pytest.raises(ValueError, match="TEST data differs"):
        evaluate_bundle(root, prepared, tmp_path / "changed_test")
    payload = read_json(root / "bundle.json")
    payload["artifacts"].pop("encoder.npz")
    write_json(root / "bundle.json", payload)
    with pytest.raises(ValueError, match="Incomplete bundle"):
        Pipeline(root)


def test_search_uses_validation_and_exports_config(tiny_config, tmp_path, monkeypatch):
    import iiot_ids.search as search_module

    calls = []

    def fake_fit(config, out):
        calls.append(config)
        return {"validation_macro_f1": 0.5}

    monkeypatch.setattr(search_module, "fit", fake_fit)
    out = tmp_path / "search"
    search_module.search(tiny_config, out, trials=2)
    assert len(calls) == 2
    assert read_json(out / "selected.json")["test_evaluated"] is False
    assert 64 <= calls[0]["model"]["width"] <= 128
    assert calls[0]["training"]["epochs"] == 2
