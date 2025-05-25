import torch
from transformer_lens import HookedTransformer

from clt import InferenceCLT
from attribution import get_vals, build_graph, prune_graph, graph_stats

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

print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
orig_stats = graph_stats(G)
print(f"Node types: {orig_stats['node_types']}")

# Prune the graph
G_pruned = prune_graph(G, logits)

print(f"\nPruned graph: {G_pruned.number_of_nodes()} nodes, {G_pruned.number_of_edges()} edges")
print(f"Node reduction: {G.number_of_nodes() / G_pruned.number_of_nodes():.1f}x")
print(f"Edge reduction: {G.number_of_edges() / G_pruned.number_of_edges():.1f}x")

pruned_stats = graph_stats(G_pruned)
print(f"\nPruned node types: {pruned_stats['node_types']}")
print(f"Pruned edge types: {pruned_stats['edge_types']}")
