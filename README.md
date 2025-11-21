# Circuit Sparsity Visualizer and Models

Tools for inspecting sparse circuit models from [Gao et al. 2025](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/). Provides code 
for running inference as well as a Streamlit dashboard that allows you to interact
with task-specific circuits found by pruning. Note: this README was AI-generated and lightly edited.

## Installation

```bash
pip install -e .
```

## Launching the Visualizer

Start the Streamlit app from the project root:

```bash
streamlit run circuit_sparsity/viz.py
```

The app loads data from the openaipublic webpage and caches locally. When the
visualizer loads you can choose a model, dataset, pruning sweep, and node budget `k`
using the controls in the left column. The plots are rendered with Plotly; most
elements are interactive and support hover/click exploration.

Example view of the Streamlit circuit visualizer (wte/wpe tab) with node ablation deltas and activation previews:

![Streamlit circuit visualizer](annotated-circuit-sparsity-viz.png)

## Running Model Forward Passes

Transformer definitions live in `circuit_sparsity.inference.gpt`. The module
exports:

- `GPTConfig` / `GPT`: lightweight GPT implementation suitable for CPU/GPU
  inference.
- `load_model(model_dir, cuda=False)`: convenience loader that expects the
  `beeg_config.json` and `final_model.pt` pair found in `models/...`.

Example usage (adapted from `tests/test_gpt.py`):

```python
from circuit_sparsity.inference.gpt import GPT, GPTConfig, load_model
from circuit_sparsity.inference.hook_utils import hook_recorder
from circuit_sparsity.registries import MODEL_BASE_DIR

config = GPTConfig(block_size=8, vocab_size=16, n_layer=1, n_head=1, d_model=8)
model = GPT(config)
logits, loss, _ = model(idx, targets=targets)

# to get activations
with hook_recorder() as rec:
    model(idx)

# rec is a dict that looks like {"0.attn.act_in": tensor(...), ...}

pretrained = load_model(f"{MODEL_BASE_DIR}/models/<model_name>", cuda=False)
```

Run tests with:

```bash
pytest tests/test_gpt.py
```

## Data Layout

Project assets live under `https://openaipublic.blob.core.windows.net/circuit-sparsity` with the following structure:

- `models/<model_name>/`
  - `beeg_config.json`: serialized `GPTConfig` used to rebuild the model.
  - `final_model.pt`: checkpoint used by `circuit_sparsity.inference.gpt.load_model`.
- `viz/<experiment>/<model_name>/<task_name>/<sweep>/<k>/`
  - `viz_data.pkl`: primary payload loaded by `viz.py` (contains circuit masks,
    activations, samples, importances, etc.).
  - Additional per-run outputs (masks, histograms, sample buckets) are stored
    under the same tree when produced by the preprocessing scripts.
- `train_curves/<model_name>/progress.json`: training metrics consumed by
  the dashboard’s summary table.
- Other experiment-specific directories (for example
  `csp_yolo1/`, `csp_yolo2/`) hold raw artifacts
  produced while preparing pruning runs.

The file paths surfaced in `viz.py` and `registries.py` assume this layout.
Update `registries.py` if you relocate the data.

## Models

We release all of the models used to obtain the results in the paper. See `registries.py` for a list of all models. Exact training hyperparameters can be found in ...

- `csp_yolo1`: This is the model used in the `single_double_quote` qualitative results. This is a 118M total param model. This is a somewhat older model that was trained with methods not exactly the same as in the paper.
- `csp_yolo2`: This is the model used in the `bracket_counting` and `set_or_string_fixedvarname` qualitative results. This is a 475M total param model.
- `csp_sweep1_*`: These models are used to obtain the figure 3 results.

## Additional Utilities

- `per_token_viz_demo.py`: minimal examples for token-level visualizations.

The project relies on Streamlit, Plotly, matplotlib, seaborn, and torch (see
`pyproject.toml` for the full dependency list).
