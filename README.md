This repo trains Cross-Layer Transcoders and creates/prunes attribution graphs as detailed by [Ameisen et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Attribution graph visualization is in progress.

## Usage
Clone the repo and run `uv sync` in its directory. Run `train.py` like so:
```bash
uv run torchrun
    --nproc_per_node=4 \
    train.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "HuggingFaceFW/fineweb" \
    --n_toks 300000000 \
    --bs 16 \
    --n_features 450048
```
`train.py` contains a more detailed config specifying more hyperparameters.
