import json
import os
import random
from dataclasses import dataclass, replace
from typing import Callable

import tiktoken
import torch
import torch.nn.functional as F
from idemlib import CacheHelper
from tiktoken.load import read_file_cached

MODEL_BASE_DIR = "https://openaipublic.blob.core.windows.net/circuit-sparsity"


@dataclass
class Datapoint:
    inputs: torch.Tensor
    patch_from_inputs: torch.Tensor | None = None
    # for scrubbing (deprecated)
    pretrain_inputs: torch.Tensor | None = None

    target_tokid: int | None = None
    target_alt_tokid: int | None = None
    # def __post_init__(self):
    #     assert self.inputs.shape[0] == 1
    #     assert len(self.inputs.shape) == 2


def last_token_contrast_loss_fn(
    x: Datapoint,
    logits: torch.Tensor,
    *,
    tokid: int | None = None,
    two_sided: bool = True,
    alt_tokid: int | None = None,
) -> torch.Tensor:
    assert two_sided, "other code path is dead and a bad idea"

    if (
        x.target_tokid is not None
    ):  # hacky, we should eventually get rid of this in a refactor
        assert tokid is None
        assert x.target_tokid is not None
        tokid = x.target_tokid
    if x.target_alt_tokid is not None:
        assert alt_tokid is None
        assert x.target_alt_tokid is not None
        alt_tokid = x.target_alt_tokid

    if two_sided:
        logits_patched, logits_clean = logits
    else:
        logits_patched = logits

    if alt_tokid is not None:
        logps_patched = F.log_softmax(logits_patched[:, [tokid, alt_tokid]], dim=-1)
        logps_clean = F.log_softmax(logits_clean[:, [tokid, alt_tokid]], dim=-1)
        return -(logps_patched[-1, 0] + (logps_clean[-1, 1] if two_sided else 0))
        # acc = (logits_patched[-1, tokid] > logits_patched[-1, alt_tokid]).float()
        # if two_sided:
        #     acc += (logits_clean[-1, tokid] < logits_clean[-1, alt_tokid]).float()
        #     acc /= 2

        # return logits_patched[-1, alt_tokid] - logits_patched[-1, tokid]

    logps_patched = F.log_softmax(logits_patched, dim=-1)
    logps_clean = F.log_softmax(logits_clean, dim=-1)

    loss = -(
        logps_patched[-1, tokid]
        + (
            torch.logsumexp(
                torch.cat(
                    [
                        logps_clean[-1, :tokid],
                        logps_clean[-1, tokid + 1 :],
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
            if two_sided
            else 0
        )
    )
    return loss


@dataclass
class Task:
    xs: list[Datapoint]
    xs_test: list[Datapoint]
    loss_fn: Callable
    enc: tiktoken.Encoding

    def __post_init__(self):
        rnd = random.Random(0)
        rnd.shuffle(self.xs)
        rnd.shuffle(self.xs_test)

    def limit(self, n: int):
        return Task(
            xs=self.xs[:n],
            xs_test=self.xs_test[:n],
            loss_fn=self.loss_fn,
            enc=self.enc,
        )

    def left_truncate_to_len(self, n: int = 255):
        return Task(
            xs=[
                replace(
                    dp,
                    inputs=dp.inputs[:, -n:],
                    patch_from_inputs=dp.patch_from_inputs[:, -n:]
                    if dp.patch_from_inputs is not None
                    else None,
                    pretrain_inputs=dp.pretrain_inputs[:, -n:]
                    if dp.pretrain_inputs is not None
                    else None,
                )
                for dp in self.xs
            ],
            xs_test=self.xs_test,
            loss_fn=self.loss_fn,
            enc=self.enc,
        )


def make_retokenizer(enc: tiktoken.Encoding, old_enc: tiktoken.Encoding):
    def _maybe_untensorify(f):
        def _f(x):
            if isinstance(x, torch.Tensor):
                return torch.tensor(f(x.tolist()))
            else:
                return f(x)

        return _f

    @_maybe_untensorify
    def _f(x):
        if isinstance(x[0], list):
            return [_f(x) for x in x]
        else:
            return enc.encode(old_enc.decode(x))

    return _f


blob_cache = CacheHelper(os.path.expanduser("~/data/cs/idemlib"))


def list_join(xss: list[list]) -> list:
    """monadic join for lists"""
    return [x for xs in xss for x in xs]


def keymap(fs: dict[str, Callable], d: dict) -> dict:
    assert set(fs.keys()).issubset(d.keys())
    return {k: fs.get(k, lambda x: x)(v) for k, v in d.items()}


def load_jsonls(paths):
    ret = []
    for path in paths:
        b = read_file_cached(path)
        ret.extend([json.loads(x) for x in b.decode().splitlines()])

    return ret


def pad_or_clip_to_len(x, n, v):
    assert len(x.shape) == 1
    # clip from the left
    if x.shape[0] > n:
        res = x[-n:]
    else:
        res = torch.cat(
            [
                torch.full((n - x.shape[0],), v, device=x.device),
                x,
            ]
        )

    return res


def only_one(xs):
    assert len(xs) == 1
    return xs[0]


def get_xs_simple(dset_obs):
    return [
        Datapoint(
            inputs=torch.tensor(dset_ob["tokens"]).cuda().unsqueeze(0),
        )
        for dset_ob in dset_obs
    ]


def lcompose(*fs):
    def _f(x):
        for f in fs:
            x = f(x)
        return x

    return _f
