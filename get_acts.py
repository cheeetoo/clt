import math
import torch
import numpy as np
from datasets import load_dataset
from einops import rearrange
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONF = "sample-10BT"
N_TOKENS = 300_000_000
BATCH_SIZE = 1024
OUT_PATH = "acts.bin"
META_PATH = "acts_meta.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
view_dtype = "uint16"

print(f"Loading model {MODEL_NAME} on {device}")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=dtype)  # type: ignore
model.cfg.n_ctx = model.cfg.n_ctx // 32

n_layers, d_model = model.cfg.n_layers, model.cfg.d_model

shape = (N_TOKENS, n_layers, d_model, 2)
item_nbytes = dtype.itemsize
fsize = math.prod(shape) * item_nbytes
print(f"Pre-allocating {fsize / 1e9:.2f} GB memmap at {OUT_PATH}")
with open(OUT_PATH, "wb") as f:
    f.truncate(fsize)
bin_mmap = np.memmap(OUT_PATH, mode="r+", dtype=np.dtype(view_dtype), shape=shape)

print(f"Loading dataset {DATASET_NAME}-{DATASET_CONF}")
written = 0
dataset = load_dataset(DATASET_NAME, DATASET_CONF, split="train", streaming=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)  # type: ignore

for batch in tqdm(dataloader):
    toks = model.to_tokens(batch["text"], truncate=True)
    n_toks = toks.shape[0] * toks.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda n: n.endswith("normalized") or n.endswith("mlp_out"),
        )
    pre = torch.stack([cache["normalized", i, "ln2"] for i in range(n_layers)])
    post = torch.stack([cache["mlp_out", i] for i in range(n_layers)])
    pre = rearrange(pre, "l b t d -> (b t) l d")
    post = rearrange(post, "l b t d -> (b t) l d")

    out = (
        torch.stack((pre, post), dim=-1)
        .to(dtype)
        .view(getattr(torch, view_dtype))
        .cpu()
    )
    if written + n_toks > N_TOKENS:
        n_toks = N_TOKENS - written
        out = out[:n_toks]
    bin_mmap[written : written + n_toks] = out.numpy()
    written += n_toks

    if written == N_TOKENS:
        break

print(f"Wrote {written} tokens to disk")
torch.save(
    dict(shape=shape, logical_dtype=dtype, storage_dtype=np.dtype(view_dtype)),
    META_PATH,
)
