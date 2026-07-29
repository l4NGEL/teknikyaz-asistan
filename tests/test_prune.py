import pytest
import torch

from src.optimize.prune import magnitude_prune, sparsity_report


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )


def test_pruned_model_has_higher_sparsity_than_original():
    model = _tiny_model()
    pruned = magnitude_prune(model, amount=0.3)

    before = sparsity_report(model)
    after = sparsity_report(pruned)

    assert after["sparsity"] > before["sparsity"]
    assert after["sparsity"] == pytest.approx(0.3, abs=0.02)


def test_original_model_is_not_mutated():
    model = _tiny_model()
    original_weight = model[0].weight.clone()

    magnitude_prune(model, amount=0.5)

    assert torch.equal(model[0].weight, original_weight)
