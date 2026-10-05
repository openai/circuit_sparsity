"""Minimal kernel implementations used by :mod:`circuit_sparsity.inference.gpt`.

The production ``circuit_sparsity.kernels`` module integrates with a bespoke set
of CUDA kernels.  Those dependencies are unavailable in the standalone
inference-only build, so we provide light-weight fallbacks that mimic the parts
of the API exercised by :class:`~circuit_sparsity.inference.gpt.GPT`.
"""

from __future__ import annotations

import math
import warnings
from functools import partial

import torch
import torch.nn as nn


def sample_top_k(n: int, k: int, shape: tuple[int, ...]) -> torch.Tensor:
    del shape  # kept for API compatibility
    x = torch.randn(n)
    inds = torch.topk(x.abs(), k=k).indices
    vals = x[inds]
    return vals


def dense_to_sparse_topk(x, k, scores=None, abs: bool = True, exact: bool = False):
    if scores is None:
        scores = x.detach()
    assert scores.shape == x.shape
    _, inds = topk(scores, k, abs=abs, exact=exact)
    # Scores select positions; values and their gradients come from the input.
    vals = x.reshape(-1)[inds]

    ret = torch.sparse_coo_tensor(
        torch.stack([inds // x.shape[1], inds % x.shape[1]], dim=0), vals, x.shape
    )
    assert ret._nnz() == k, f"{ret._nnz()} != {k}"
    return ret


def topk(x, k, abs: bool = False, exact: bool = True, dim=None):
    if k >= x.numel():
        k = x.numel()

    if exact:
        topk_fn = partial(torch.topk, sorted=False)
    else:
        raise NotImplementedError("approximate topk is not available in torn out build")

    if dim is None:
        x = x.flatten()
        topk_fn = partial(topk_fn, dim=0)
    else:
        assert dim in [0, 1]
        assert k % x.shape[1 - dim] == 0, f"{k} % {x.shape[1 - dim]} != 0"
        k //= x.shape[1 - dim]
        assert len(x.shape) == 2, "todo: generalize to higher dims"

        def _topk_fn(x, k, topk_fn):
            vals, inds = topk_fn(x, k, dim=dim)
            if dim == 0:
                inds = inds * x.shape[1] + torch.arange(
                    x.shape[1], device=x.device, dtype=inds.dtype
                ).unsqueeze(0)
            else:
                inds = (
                    inds
                    + torch.arange(x.shape[0], device=x.device, dtype=inds.dtype).unsqueeze(1)
                    * x.shape[1]
                )

            inds = inds.flatten()
            vals = vals.flatten()
            return vals, inds

        topk_fn = partial(_topk_fn, topk_fn=topk_fn)

    if abs:
        _, inds = topk_fn(x.abs(), k)
        vals = x.flatten()[inds]
    else:
        vals, inds = topk_fn(x, k)

    return vals, inds


def coo_dense_sparse_T_matmul(dense: torch.Tensor, sparse_T: torch.Tensor, impl: str):
    # mock implementation
    return naive_torch_coo_dense_sparse_T_matmul(dense, sparse_T)


def naive_torch_coo_dense_sparse_T_matmul(dense: torch.Tensor, sparse_T: torch.Tensor) -> torch.Tensor:
    dense = dense.contiguous()
    sparse_T = sparse_T.coalesce()
    result = (dense.float() @ sparse_T.T.float()).to(dense.dtype)
    return result
