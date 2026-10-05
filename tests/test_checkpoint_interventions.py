"""CPU checks for transformed interventions across checkpoint recomputation."""

import pytest
import torch

from circuit_sparsity.inference.hook_utils import (
    get_context,
    hook_intervention_transform,
    hook_namespace,
    hook_recorder,
    hook_save,
    torch_recompute_preserving_hook_context,
)


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("after_scope", [False, True])
def test_checkpoint_preserves_transformed_interventions(reentrant, recording, nested, after_scope):
    from contextlib import ExitStack

    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, requires_grad=True)
    context = get_context()
    original_transform = context.curintervtransformer
    recorder = None

    def transformation(interventions):
        return {**interventions, "layer.hidden": lambda value: value * 3}

    def second_transformation(interventions):
        original = interventions["layer.hidden"]
        return {**interventions, "layer.hidden": lambda value: original(value) + 2}

    def function(value):
        hidden = hook_save("hidden", value.square())
        return hidden.sin()

    with ExitStack() as stack:
        if recording:
            recorder = stack.enter_context(hook_recorder())
        stack.enter_context(hook_intervention_transform(transformation))
        if nested:
            stack.enter_context(hook_intervention_transform(second_transformation))
        stack.enter_context(hook_namespace("layer"))
        actual = torch_recompute_preserving_hook_context(function, x, use_reentrant=reentrant)
        expected_hidden = x.square() * 3 + (2 if nested else 0)
        expected = expected_hidden.sin()
        torch.testing.assert_close(actual, expected)
        if recorder is not None:
            torch.testing.assert_close(recorder["layer.hidden"], expected_hidden)
        assert get_context() is context
        if not after_scope:
            actual.sum().backward()
    if after_scope:
        actual.sum().backward()
    torch.testing.assert_close(x.grad, 6 * x.detach() * expected_hidden.detach().cos())
    assert get_context() is context
    assert context.curintervtransformer is original_transform
    assert context.curcontext is None


@pytest.mark.parametrize("reentrant", [False, True])
def test_plain_checkpoint_values_and_gradients_are_unchanged(reentrant):
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    result = torch_recompute_preserving_hook_context(
        lambda value: value.square(), x, use_reentrant=reentrant
    )
    torch.testing.assert_close(result, x.square())
    result.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())


def test_transform_and_context_restore_after_exception():
    context = get_context()
    original = context.curintervtransformer

    def fail(value):
        hook_save("hidden", value.square())
        raise ValueError("Synthetic checkpoint failure")

    with pytest.raises(ValueError, match="Synthetic checkpoint failure"):
        with hook_intervention_transform(lambda values: {**values, "hidden": lambda x: x * 3}):
            torch_recompute_preserving_hook_context(
                fail, torch.tensor([1.0], requires_grad=True), use_reentrant=False
            )
    assert get_context() is context
    assert context.curintervtransformer is original
