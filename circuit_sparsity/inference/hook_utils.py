"""Self-contained subset of :mod:`circuit_sparsity.hook_utils` for inference builds.

The full module has no exotic dependencies, but mirroring the definitions here
keeps the trimmed :mod:`circuit_sparsity.inference.gpt` module hermetic and easy to vendor.  The
implementations below are copied with minor tweaks for readability so that code
written against :func:`hook_recorder`, :func:`hook_namespace`, and
:func:`torch_recompute_preserving_hook_context` behaves identically in both the
training and inference configurations.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from functools import partial

import torch
import torch.utils.checkpoint


class HookContext:
    """State container used by the hook helpers."""

    def __init__(self) -> None:
        self._reset()
        self.curintervtransformer = lambda x: x

    def _reset(self) -> None:
        self.curcontext = None
        self.curname = ""
        self.curregex = None
        self.curinterventions = None
        self.save_grads = None

    def _get_interventions(self):
        return self.curintervtransformer(
            self.curinterventions if self.curinterventions is not None else {}
        )

    @contextmanager
    def hook_recorder(self, regex: str = ".*", interventions=None, save_grads: bool = False):
        """Record tensors that pass through hooks matching ``regex``."""

        assert self.curcontext is None, "reentrancy not allowed!"

        try:
            self.curcontext = {}
            self.curregex = re.compile(regex)
            self.curname = ""
            self.curinterventions = interventions
            self.save_grads = save_grads

            yield self.curcontext
        finally:
            self._reset()
            get_context()._reset()

    @contextmanager
    def hook_intervention_transform(self, intervention_transformer):
        oldintervention_transformer = self.curintervtransformer

        def compose(f, g):
            return lambda x: f(g(x))

        self.curintervtransformer = compose(
            intervention_transformer,
            self.curintervtransformer,
        )

        try:
            yield
        finally:
            self.curintervtransformer = oldintervention_transformer

    @contextmanager
    def hook_namespace(self, name: str):
        """Temporarily push ``name`` onto the hook namespace stack."""

        oldname = self.curname
        self.curname = self.curname + name + "."

        try:
            yield
        finally:
            self.curname = oldname

    def hook_save(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Optionally record ``tensor`` using the current namespace."""

        curinterventions = self._get_interventions()
        if curinterventions is not None:
            key = self.curname + name
            if key in curinterventions:
                tensor = curinterventions[key](tensor)

        if self.curcontext is not None and self.curregex.match(self.curname + name):
            self.curcontext[self.curname + name] = tensor

        if self.curcontext is not None and self.save_grads and tensor.requires_grad:

            class _Grad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input_tensor):
                    return input_tensor

                @staticmethod
                def backward(ctx, grad_output):
                    self.curcontext[self.curname + name + ".grad"] = grad_output
                    return grad_output

            if self.curregex.match(self.curname + name + ".grad"):
                tensor = _Grad.apply(tensor)

        return tensor


def set_context(new_context: HookContext) -> None:
    global context
    context = new_context


def get_context() -> HookContext:
    global context
    return context


def torch_recompute_preserving_hook_context(f, *xs, use_reentrant=None):
    """Wrapper around :func:`torch.utils.checkpoint` that propagates hooks."""

    oldcontext = get_context()
    curcontext = HookContext()
    curcontext.curcontext = (
        dict(oldcontext.curcontext) if oldcontext.curcontext is not None else None
    )
    curcontext.curregex = oldcontext.curregex
    curcontext.curname = oldcontext.curname
    curcontext.curinterventions = (
        dict(oldcontext.curinterventions) if oldcontext.curinterventions is not None else None
    )
    curcontext.save_grads = oldcontext.save_grads
    curcontext.curintervtransformer = oldcontext.curintervtransformer

    is_recompute = False

    def _f(curcontext: HookContext, *xs):
        initcontext = get_context()
        nonlocal is_recompute

        set_context(curcontext)
        try:
            res = f(*xs)

            if not is_recompute and oldcontext.curcontext is not None:
                oldcontext.curcontext |= curcontext.curcontext
        finally:
            set_context(initcontext)
            is_recompute = True
        return res

    res = torch.utils.checkpoint.checkpoint(
        partial(_f, curcontext), *xs, use_reentrant=use_reentrant
    )

    return res


context = HookContext()


def hook_recorder(*a, **k):
    return get_context().hook_recorder(*a, **k)


def hook_namespace(*a, **k):
    return get_context().hook_namespace(*a, **k)


def hook_save(*a, **k):
    return get_context().hook_save(*a, **k)


def hook_intervention_transform(*a, **k):
    return get_context().hook_intervention_transform(*a, **k)


