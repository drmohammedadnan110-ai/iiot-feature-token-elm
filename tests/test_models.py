import numpy as np
import pytest
import torch
from torch.nn import functional as F

from iiot_ids.baselines import make_estimator
from iiot_ids.elm import RidgeELM
from iiot_ids.model import FeatureEncoder, TrainingModel, focal_loss, inverse_frequency_weights


@pytest.mark.parametrize("attention", [True, False])
def test_feature_tokens_and_gradient(attention):
    encoder = FeatureEncoder(2, [3], width=8, heads=2, layers=1, attention=attention)
    model = TrainingModel(encoder, 3)
    output = model(torch.randn(4, 2), torch.tensor([[0], [1], [2], [1]]))
    assert output.shape == (4, 3)
    output.square().mean().backward()
    assert encoder.tokenizer.identity.grad is not None
    assert encoder.tokenizer.numeric_weight.grad.abs().sum() > 0


def test_focal_zero_gamma_and_weights():
    logits = torch.tensor([[2.0, -1.0], [0.0, 1.0], [3.0, -2.0]])
    labels = torch.tensor([0, 1, 1])
    weights = inverse_frequency_weights(labels, 2)
    expected = (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).mean()
    assert focal_loss(logits, labels, weights, gamma=0) == pytest.approx(expected.item())
    with pytest.raises(ValueError):
        inverse_frequency_weights([0, 0], 2)


@pytest.mark.parametrize("initialization", ["gaussian", "orthogonal", "trigonometric"])
@pytest.mark.parametrize("activation", ["gelu", "relu", "tanh"])
def test_streamed_ridge_matches_independent_augmented_lstsq(initialization, activation, tmp_path):
    rng = np.random.default_rng(21)
    z, y = rng.normal(size=(30, 4)), np.arange(30) % 3
    model = RidgeELM(4, 3, hidden=9, ridge=0.2, initialization=initialization, activation=activation, seed=7)
    model.fit_batches([(z[:13], y[:13]), (z[13:], y[13:])])
    h = model.features(z)
    augmented_x = np.row_stack([h, np.sqrt(0.2) * np.eye(9)])
    augmented_y = np.row_stack([np.eye(3)[y], np.zeros((9, 3))])
    reference = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    np.testing.assert_allclose(model.beta, reference, rtol=1e-8, atol=1e-10)
    duplicate = RidgeELM(
        4, 3, hidden=9, ridge=0.2, initialization=initialization, activation=activation, seed=7
    )
    duplicate.fit_batches([(z, y)])
    np.testing.assert_allclose(model.predict(z), duplicate.predict(z), atol=1e-10)
    model.save(tmp_path)
    np.testing.assert_allclose(RidgeELM.load(tmp_path).predict(z), model.predict(z))


def test_ridge_rejects_zero_and_baseline_overrides():
    with pytest.raises(ValueError):
        RidgeELM(3, 2, ridge=0)
    model = make_estimator("logistic", 7, {"max_iter": 25, "random_state": 99})
    assert model.max_iter == 25 and model.random_state == 7
    assert make_estimator("extra_trees", 7, {"n_jobs": 2}).n_jobs == 2
