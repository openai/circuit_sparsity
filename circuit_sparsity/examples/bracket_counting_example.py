"""
Reproduce bracket-counting mismatch on csp_yolo2: model predicts ']' instead of ']]' on the first viz sample.
Run from repo root:
  source .venv/bin/activate
  GAO_ONLINE=1 python circuit_sparsity/repro_bracket_mismatch.py
"""

from __future__ import annotations

import io
import os

import torch
from tiktoken import Encoding
from tiktoken.load import read_file_cached

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def _truncate_zeros(sample: torch.Tensor) -> torch.Tensor:
    # remove trailing zeros (used as padding tokens)
    assert sample.ndim == 1
    non_zero_indices = sample.nonzero()
    if non_zero_indices.numel() == 0:
        return sample
    return sample[: non_zero_indices[-1] + 1]


def main() -> None:
    enc = Encoding(**tinypython.tinypython_2k())
    model_name = "csp_yolo2"
    model_path = os.path.expanduser(f"{MODEL_BASE_DIR}/models/{model_name}")
    model = load_model(model_path, flash=True, cuda=False)

    viz_path = os.path.expanduser(
        f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
    )
    buf = io.BytesIO(read_file_cached(viz_path))
    viz = torch.load(buf, map_location="cpu", weights_only=True)
    sample = _truncate_zeros(viz["importances"]["task_samples"][0][0])

    tok_double = enc.encode("]]\n")[0]
    tok_single = enc.encode("]\n")[0]

    with torch.no_grad():
        print(f"{sample=}")
        logits, _, _ = model(sample.unsqueeze(0))
        last = logits[0, -1]
        log_double = float(last[tok_double])
        log_single = float(last[tok_single])
        pred = tok_double if log_double > log_single else tok_single

    text = enc.decode(sample.tolist())
    print("decoded sample:\n", text)
    print("logits ]] / ]", log_double, log_single)
    print("pred token id", pred, "decoded", enc.decode([pred]))


if __name__ == "__main__":
    main()