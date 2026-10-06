import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from iiot_ids.generalization import unknown_metrics
from iiot_ids.metrics import evaluate, fit_temperature, probabilities
from iiot_ids.statistics import holm, paired_group_bootstrap, paired_seed_tests


def test_temperature_and_fixed_classes():
    scores, y = np.array([[8.0, 0], [0, 8], [8, 0], [0, 8]]), np.array([0, 1, 1, 1])
    temperature = fit_temperature(scores, y)
    assert temperature > 0
    p = probabilities(scores, temperature)
    assert (p.argmax(1) == scores.argmax(1)).all()
    loss = lambda s: (logsumexp(s, axis=1) - s[np.arange(4), y]).mean()
    assert loss(scores / temperature) <= loss(scores)
    result = evaluate([0, 0], [[1.0, 0, 0], [0.2, 0.3, 0.5]], ["Normal", "a", "b"])
    assert result["balanced_accuracy"] == pytest.approx(1 / 6)
    assert result["macro_roc_auc"] is None
    assert sum(b["n"] for b in result["bins"]) == 2
    assert result["binary_argmax_collapse"]["fp"] == 1


def test_paired_statistics_alignment_and_population():
    rows = [
        {
            "seed": s,
            "model": model,
            "macro_f1": 0.8 + s * 0.01 + improvement,
            "test_sha256": "shared",
            "classes": json.dumps(["a", "b"]),
        }
        for s in range(3)
        for model, improvement in [("reference", 0.02), ("baseline", 0)]
    ]
    frame = pd.DataFrame(rows)
    result = paired_seed_tests(frame, "reference")
    assert result["comparisons"]["baseline"]["mean"] == pytest.approx(0.02)
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    for key, value in [("test_sha256", "other"), ("classes", "other"), ("seed", 15)]:
        broken = frame.copy()
        broken.loc[0, key] = value
        with pytest.raises(ValueError):
            paired_seed_tests(broken, "reference")


def test_identical_group_bootstrap_and_labels():
    a = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d"],
            "group": ["g1", "g1", "g2", "g2"],
            "seed": [7] * 4,
            "y_true": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
        }
    )
    result = paired_group_bootstrap(a, a, ["a", "b"], draws=20)
    assert result["macro_f1"]["ci95_difference"] == [0.0, 0.0]
    b = a.copy()
    b.loc[0, "y_true"] = 1
    with pytest.raises(ValueError):
        paired_group_bootstrap(a, b, ["a", "b"], draws=20)


def test_unknown_threshold_independent_of_test_labels():
    validation = [0.1, 0.2, 0.3, 0.4, 0.5]
    a = unknown_metrics(validation, [0.2, 0.8, 0.9, 0.3], [False, True, True, False], 0.2)
    b = unknown_metrics(validation, [0.2, 0.8, 0.9, 0.3], [True, False, False, True], 0.2)
    assert a["threshold"] == b["threshold"] == 0.5
    assert a["unknown_recall"] == 1 and a["known_false_unknown_rate"] == 0
