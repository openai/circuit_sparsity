"""CPU regressions for gradient ownership in recorded activation hooks."""

import pytest
import torch

from circuit_sparsity.inference.hook_utils import (
    get_context,
    hook_namespace,
    hook_recorder,
    hook_save,
)


@pytest.mark.parametrize("after_recorder", [False, True])
@pytest.mark.parametrize("gradient_only", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_keys_keep_forward_namespaces(after_recorder, gradient_only, dtype):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, requires_grad=True)
    pattern = r".*\.grad$" if gradient_only else ".*"
    with hook_recorder(regex=pattern, save_grads=True) as recorded:
        with hook_namespace("model"):
            with hook_namespace("left"):
                left = hook_save("activation", x * 2)
            with hook_namespace("right"):
                right = hook_save("activation", x * 3)
        loss = (left * 5 + right * 7).sum()
        if not after_recorder:
            loss.backward()
    if after_recorder:
        loss.backward()
    expected_keys = {"model.left.activation.grad", "model.right.activation.grad"}
    if not gradient_only:
        expected_keys |= {"model.left.activation", "model.right.activation"}
    assert set(recorded) == expected_keys
    torch.testing.assert_close(recorded["model.left.activation.grad"], torch.full_like(x, 5))
    torch.testing.assert_close(recorded["model.right.activation.grad"], torch.full_like(x, 7))
    torch.testing.assert_close(x.grad, torch.full_like(x, 31))
    assert get_context().curcontext is None


def test_old_backward_does_not_write_into_a_new_recorder():
    x = torch.tensor([2.0], requires_grad=True)
    with hook_recorder(save_grads=True) as first:
        with hook_namespace("old"):
            value = hook_save("activation", x * 3)
    with hook_recorder(save_grads=True) as second:
        with hook_namespace("new"):
            other = hook_save("activation", x * 4)
        value.sum().backward(retain_graph=True)
        assert "old.activation.grad" in first
        assert set(second) == {"new.activation"}
        other.sum().backward()
    assert set(first) == {"old.activation", "old.activation.grad"}
    assert set(second) == {"new.activation", "new.activation.grad"}
    torch.testing.assert_close(x.grad, torch.tensor([7.0]))


def test_repeated_backward_retains_latest_hook_gradient():
    x = torch.tensor([2.0], requires_grad=True)
    with hook_recorder(save_grads=True) as recorded:
        with hook_namespace("layer"):
            value = hook_save("activation", x * 3)
        value.sum().backward(retain_graph=True)
        (value * 2).sum().backward()
    torch.testing.assert_close(recorded["layer.activation.grad"], torch.tensor([2.0]))
    torch.testing.assert_close(x.grad, torch.tensor([9.0]))


@pytest.mark.parametrize("save_grads,requires_grad", [(False, True), (True, False), (False, False)])
def test_disabled_recording_does_not_add_gradient_entries(save_grads, requires_grad):
    x = torch.tensor([2.0], requires_grad=requires_grad)
    with hook_recorder(save_grads=save_grads) as recorded:
        with hook_namespace("layer"):
            value = hook_save("activation", x * 3)
        if requires_grad:
            value.sum().backward()
    assert set(recorded) == {"layer.activation"}
    if requires_grad:
        torch.testing.assert_close(x.grad, torch.tensor([3.0]))


def test_excluded_gradient_is_not_recorded():
    x = torch.tensor([2.0], requires_grad=True)
    with hook_recorder(regex=r"layer\.activation$", save_grads=True) as recorded:
        with hook_namespace("layer"):
            value = hook_save("activation", x * 3)
        value.sum().backward()
    assert set(recorded) == {"layer.activation"}
    torch.testing.assert_close(x.grad, torch.tensor([3.0]))
