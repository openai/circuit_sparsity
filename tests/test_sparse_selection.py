"""Offline regression coverage for sparse value selection and autograd."""

import pytest
import torch
from circuit_sparsity.inference.kernels import dense_to_sparse_topk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [0, 1, 3, 6])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("custom_scores", [False, True])
def test_topk_retains_input_values_and_gradients(dtype, k, absolute, custom_scores):
    x = torch.tensor([[0.1, -0.6, 0.3], [-0.4, 0.2, 0.5]], dtype=dtype, requires_grad=True)
    scores = (
        torch.tensor([[3.0, -6.0, 2.0], [5.0, -4.0, 1.0]], dtype=dtype, requires_grad=True)
        if custom_scores
        else None
    )
    ranking = (scores if scores is not None else x).detach().flatten()
    indices = torch.topk(ranking.abs() if absolute else ranking, k, sorted=False).indices
    mask = torch.zeros(x.numel(), dtype=dtype)
    mask[indices] = 1
    mask = mask.reshape(x.shape)
    actual = dense_to_sparse_topk(x, k, scores=scores, abs=absolute, exact=True)
    assert actual.layout == torch.sparse_coo and actual._nnz() == k
    assert actual.dtype == x.dtype and actual.device == x.device
    torch.testing.assert_close(actual.to_dense(), x * mask)
    weights = torch.arange(1, 7, dtype=dtype).reshape(x.shape)
    assert actual.requires_grad, "Selected values must keep their connection to the input"
    (actual.to_dense() * weights).sum().backward()
    torch.testing.assert_close(x.grad, weights * mask)
    if scores is not None:
        assert scores.grad is None, "Scores select positions; they are not sparse tensor values"


@pytest.mark.parametrize("custom_scores", [False, True])
def test_noncontiguous_input_uses_logical_flat_indices(custom_scores):
    base = torch.arange(12.0, requires_grad=True)
    x = base.reshape(3, 4).T
    scores = -x.detach() if custom_scores else None
    actual = dense_to_sparse_topk(x, 4, scores=scores, abs=False, exact=True)
    indices = torch.topk((scores if scores is not None else x).flatten(), 4, sorted=False).indices
    mask = torch.zeros_like(x).flatten()
    mask[indices] = 1
    expected = x * mask.reshape(x.shape)
    torch.testing.assert_close(actual.to_dense(), expected)
    actual.to_dense().sum().backward()
    torch.testing.assert_close(base.grad, mask.reshape(x.shape).T.reshape(-1))


@pytest.mark.parametrize("score_dtype", [torch.float32, torch.float64])
def test_scoring_dtype_does_not_change_values_dtype(score_dtype):
    x = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=torch.float32)
    scores = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=score_dtype)
    result = dense_to_sparse_topk(x, 3, scores=scores, exact=True)
    assert result.dtype == x.dtype
    torch.testing.assert_close(result.to_dense(), torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]]))


def test_sparse_matmul_uses_selected_weights():
    from circuit_sparsity.inference.kernels import naive_torch_coo_dense_sparse_T_matmul

    weights = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], requires_grad=True)
    scores = torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    dense = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    sparse = dense_to_sparse_topk(weights, 3, scores=scores, exact=True)
    result = naive_torch_coo_dense_sparse_T_matmul(
        dense,
        sparse,
    )
    expected_weights = weights * torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    expected = dense @ expected_weights.T
    torch.testing.assert_close(result, expected)
    actual_grad = torch.autograd.grad(result.sum(), (weights, dense), retain_graph=True)
    reference_grad = torch.autograd.grad(expected.sum(), (weights, dense))
    for actual, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, reference)
