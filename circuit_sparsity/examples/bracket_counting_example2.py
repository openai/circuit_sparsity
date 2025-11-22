import os

import tiktoken
import torch

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR

# MODEL_NAME = "achyuta-csp-achy_agi_run-_8L_pfrac6.25e-2_lr1.02e-1"
MODEL_NAME = "csp_yolo2"


def main() -> None:
    enc2k = tiktoken.get_encoding('tinypython_2k')
    model_path = os.path.expanduser(f"{MODEL_BASE_DIR}/models/{MODEL_NAME}")
    model = load_model(model_path, flash=True, cuda=False)

    print(f"model_path={model_path}")

    print(enc2k.decode(enc2k.encode("]]\n")))
    print(enc2k.decode(enc2k.encode("]\n")))

    
    # for prefix_length in PREFIX_LENGTHS:
    strin = '''import numpy as np
from numba import njit

def energy_order(blueprints, n_min, max_iters):
    print("Finding the best order...")
    values =[[5, 3, 11, 3, 12, 5, 3, 11'''#, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12, 5, 3, 11, 3, 12'''
    print(len(enc2k.encode(strin)))
    e = model(torch.tensor(enc2k.encode(strin)).unsqueeze(0).cuda())[0][0][-1]
    a = e[enc2k.encode("]]\n")[0]]
    b = e[enc2k.encode("]\n")[0]]
    sft = torch.nn.functional.softmax(torch.stack([a, b]), dim=0)
    print("double bracket: ", sft[0].item(), "single bracket: ", sft[1].item())


if __name__ == "__main__":
    main()