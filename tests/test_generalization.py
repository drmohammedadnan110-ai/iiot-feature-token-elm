import numpy as np
import pandas as pd
import pytest

from iiot_ids.data import read_csv
from iiot_ids.engine import Pipeline, fit
from iiot_ids.generalization import leave_one_out, transfer
from iiot_ids.utils import read_json, write_json


def test_leave_out_excludes_unknown_from_fitting_and_calibration(tiny_config, tmp_path):
    out = tmp_path / "unknown"
    result = leave_one_out(tiny_config, "Attack_B", out)
    assert result["heldout_class"] == "Attack_B"
    for split in ["train", "val"]:
        assert "Attack_B" not in set(read_csv(out / "prepared" / f"{split}.csv").label)
    assert "Attack_B" not in read_json(out / "model" / "bundle.json")["classes"]
    predictions = pd.read_csv(out / "predictions.csv")
    assert predictions.is_unknown.any() and (~predictions.is_unknown).any()


def test_identity_transfer_matches_frozen_source_prediction(tiny_config, prepared, tmp_path):
    bundle = tmp_path / "source"
    fit(tiny_config, bundle)
    model = Pipeline(bundle)
    schema = model.schema
    mapping = {
        "label_column": schema.label,
        "labels": {c: c for c in schema.classes},
        "features": {
            n: {"column": n, "meaning": "identity synthetic feature", "unit": "identity"}
            for n in schema.numeric + schema.categorical
        },
    }
    path = tmp_path / "mapping.json"
    write_json(path, mapping)
    out = tmp_path / "transfer"
    transfer(bundle, prepared / "test.csv", path, out)
    expected = model.predict(read_csv(prepared / "test.csv"))
    actual = pd.read_csv(out / "predictions.csv")
    columns = [f"prob_{i}" for i in range(len(schema.classes))]
    np.testing.assert_allclose(actual[columns], expected[columns], atol=1e-7)
    assert read_json(out / "provenance.json")["target_fitting"] is False
    mapping["labels"].pop("Attack_B")
    write_json(path, mapping)
    with pytest.raises(ValueError, match="Every target label"):
        transfer(bundle, prepared / "test.csv", path, tmp_path / "bad_transfer")
