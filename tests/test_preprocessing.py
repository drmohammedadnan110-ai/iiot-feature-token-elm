import numpy as np
import pandas as pd
import pytest

from iiot_ids.data import assert_partitions, connected_groups, development_data, fingerprints, read_csv
from iiot_ids.preprocessing import Preprocessor, Schema


def test_train_only_and_unknown_categories(tmp_path):
    schema = Schema(["x", "empty"], ["c"], "label", ["a", "b"])
    train = pd.DataFrame(
        {"x": [0.0, 2.0, np.nan], "empty": [np.nan] * 3, "c": ["a", "b", ""], "split": ["train"] * 3}
    )
    with pytest.raises(ValueError, match="TRAIN"):
        Preprocessor(schema).fit(train.assign(split="val"))
    fitted = Preprocessor(schema).fit(train)
    assert fitted.state["medians"] == [1.0, 0.0]
    x, c = fitted.transform(
        pd.DataFrame({"x": [np.inf, 100.0], "empty": [np.nan, 0.0], "c": ["unseen", "a"]})
    )
    assert x[0, 0] == 0 and np.all(x[:, 1] == 0)
    assert c[:, 0].tolist() == [0, 1]
    assert fitted.state["fit_rows"] == 3
    with pytest.raises(RuntimeError):
        fitted.fit(train)
    fitted.save(tmp_path / "pre.json")
    assert Preprocessor.load(tmp_path / "pre.json").state == fitted.state


@pytest.mark.parametrize("bad", ["label", "split", "capture", "forbidden"])
def test_reject_label_and_metadata_features(bad):
    with pytest.raises(ValueError, match="leakage"):
        Schema([bad], [], "label", ["a", "b"], group_columns=["capture"], forbidden=["forbidden"])


def test_transitive_grouping_and_numeric_normalization():
    schema = Schema(["x"], [], "label", ["a", "b"], group_columns=["capture"])
    frame = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "x": ["0", "-0.0", "2", "3"],
            "capture": ["one", "two", "two", "four"],
            "split": ["train"] * 4,
        }
    )
    frame["fingerprint"] = fingerprints(frame, schema)
    frame["group"] = connected_groups(frame, schema)
    assert frame.group.iloc[:3].nunique() == 1
    assert_partitions(frame)
    frame.loc[2, "split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        assert_partitions(frame)
    frame.loc[0, "capture"] = ""
    with pytest.raises(ValueError, match="Missing values"):
        connected_groups(frame, schema)


def test_prepared_integrity_and_class_support(prepared):
    schema, data, _ = development_data(prepared)
    assert set(data) == {"train", "val"}
    for split in ["train", "val", "test"]:
        assert set(read_csv(prepared / f"{split}.csv")[schema.label]) == set(schema.classes)
    path = prepared / "schema.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="schema or manifest"):
        development_data(prepared)


@pytest.mark.parametrize("numeric,categorical", [(1, 0), (0, 1)])
def test_single_feature_types(numeric, categorical):
    schema = Schema(["x"] if numeric else [], ["c"] if categorical else [], "label", ["a", "b"])
    frame = pd.DataFrame({"x": [1, 2], "c": ["a", "b"], "split": ["train", "train"]})
    x, c = Preprocessor(schema).fit(frame).transform(frame)
    assert x.shape == (2, numeric) and c.shape == (2, categorical)
