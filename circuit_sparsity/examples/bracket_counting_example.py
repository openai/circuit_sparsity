"""
Reproduce bracket-counting mismatch on csp_yolo2: model predicts ']' instead of ']]' on the first viz sample.
Run from repo root:
  source .venv/bin/activate
  GAO_ONLINE=1 python circuit_sparsity/repro_bracket_mismatch.py
"""

from __future__ import annotations

import io
import json
import os

import torch
from tiktoken import Encoding
from tiktoken.load import read_file_cached

from circuit_sparsity.inference.gpt import GPT, GPTConfig
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


def _load_pruned_model_filtered(model_name: str):
    model_path = os.path.expanduser(f"{MODEL_BASE_DIR}/models/{model_name}")
    cfg_json = json.loads(read_file_cached(f"{model_path}/beeg_config.json").decode())
    cfg_json.setdefault("sink", False)
    cfg_json.setdefault("grad_checkpointing", False)
    print(f"{cfg_json=}")
    allowed = set(GPTConfig.__annotations__.keys())  # Filter out any keys that don't belong
    filtered = {k: v for k, v in cfg_json.items() if k in allowed}
    config = GPTConfig(**filtered)
    ckpt_path = os.path.join(model_path, "final_model.pt")
    model = GPT(config)
    sd = torch.load(io.BytesIO(read_file_cached(ckpt_path)), weights_only=True, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main() -> None:
    enc = Encoding(**tinypython.tinypython_2k())
    model = _load_pruned_model_filtered("csp_yolo2")

    viz_path = os.path.expanduser(
        f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
    )
    buf = io.BytesIO(read_file_cached(viz_path))
    viz = torch.load(buf, map_location="cpu", weights_only=True)
    sample = viz["importances"]["task_samples"][0][0]

    tok_double = enc.encode("]]\n")[0]
    tok_single = enc.encode("]\n")[0]

    with torch.no_grad():
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