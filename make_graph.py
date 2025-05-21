import torch
from transformer_lens import HookedTransformer

from clt import InferenceCLT
from attribution import get_vals, build_graph

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Federal Bureau of Investigation (F"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=dtype)  # type: ignore

sd = torch.load("model.pt", map_location=device)
clt = InferenceCLT.from_pretrained(sd).to(device)

toks, logits, acts, error, resid_nodes = get_vals(model=model, prompt=PROMPT, clt=clt)

G = build_graph(
    toks=toks,
    logits=logits,
    model=model,
    clt=clt,
    acts=acts,
    error=error,
    resid_nodes=resid_nodes,
)
