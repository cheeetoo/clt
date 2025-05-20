import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from transformer_lens import HookedTransformer, utils
import networkx as nx

from clt import InferenceCLT

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Federal Bureau of Investigation (F"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=dtype)  # type: ignore

L = model.cfg.n_layers

sd = torch.load("model.pt", map_location=device)
clt = InferenceCLT.from_pretrained(sd).to(device)

with torch.no_grad():
    toks = model.to_tokens(PROMPT)
    logits_ref, cache_ref = model.run_with_cache(  # type: ignore
        toks,
        names_filter=lambda n: (
            n.endswith("pattern")
            or n.endswith("scale")
            or n.endswith("normalized")
            or n.endswith("mlp_out")
            or n == "embed"
        ),
    )


def freeze(model, *name):
    ref = cache_ref[name].detach()

    def _hook(tensor, hook):
        return ref

    model.add_hook(utils.get_act_name(name), _hook)  # type: ignore


for i in range(L):
    freeze(model, "pattern", i)
    freeze(model, "scale", i, "ln1")
    freeze(model, "scale", i, "ln2")

pre = torch.stack([cache_ref["normalized", i, "ln2"] for i in range(L)])
post = torch.stack([cache_ref["mlp_out", i] for i in range(L)])
pre = rearrange(pre, "l b t d -> (b t) l d")
post = rearrange(post, "l b t d -> b t l d")

with torch.no_grad():
    _, acts_full, x_hat_full = clt(pre)

acts = rearrange(acts_full, "(b t) l f -> b t l f", b=toks.size(0))
x_hat = rearrange(x_hat_full, "(b t) l d -> b t l d", b=toks.size(0))

error = post - x_hat

for i in range(L):
    mlp_out = (x_hat[:, :, i] + error[:, :, i]).detach()

    def _mlp_replacer(tensor, hook):
        return mlp_out

    model.add_hook(utils.get_act_name("mlp_out", i), _mlp_replacer)

embed = model.W_E.detach().clone().requires_grad_(True)

logits, cache = model.run_with_cache(
    toks,
    names_filter=lambda n: n.endswith("normalized") or n == "embed",
    return_type=None,
)

resid_nodes = []
for i in range(L):
    cache["normalized", i, "ln2"].requires_grad_(True)
    resid_nodes.append(cache["normalized", i, "ln2"])


def _backward_to_sources(target_layer: int, target_pos: int, v_in: Tensor):
    grad_out = torch.zeros_like(resid_nodes[target_layer])
    grad_out[:, target_pos] = v_in

    grads = torch.autograd.grad(
        outputs=resid_nodes[target_layer],
        inputs=[embed] + resid_nodes,
        grad_outputs=grad_out,
        retain_graph=True,
        allow_unused=True,
    )
    grad_embed, grad_resid = grads[0], grads[1:]
    return grad_embed, grad_resid


T = toks.size(-1)
outs_prob = torch.softmax(logits[0, -1], dim=-1)  # type: ignore
vals, idxs = outs_prob.topk(10)
idx = torch.searchsorted(vals.cumsum(-1), -0.95)
idx = min(idx + 1, vals.size(-1))
outs = idxs[:idx].tolist()

G = nx.DiGraph()


def node_id(kind, *args):
    return (kind,) + tuple(args)


active_feats = []
for pos in range(T):
    for layer in range(L):
        nz = (acts[0, pos, layer] > 0).nonzero(as_tuple=True)[0]
        for f in nz.tolist():
            a_s = acts[0, pos, layer, f].item()
            active_feats.append((pos, layer, f, a_s))

# iterate over feature targets
for pos_t, l_t, f_t, _ in active_feats:
    tgt = node_id("feat", pos_t, l_t, f_t)
    v_in = clt.w_e[l_t, :, f_t].detach()
    grad_emb, grad_resid = _backward_to_sources(l_t, pos_t, v_in)
    grad_resid = [g[0] for g in grad_resid]  # TODO: remove

    # emb sources
    for pos_s in range(T):
        tok_id: int = toks[0, pos_s].item()  # type: ignore
        w = (embed[tok_id] * grad_resid[0][pos_s]).sum().item()
        if w != 0:
            G.add_edge(node_id("emb", pos_s), tgt, weight=w)

    # err sources
    for l_err in range(l_t):
        for pos_err in range(T):
            v_out = error[0, pos_err, l_err]
            w = (v_out * grad_resid[l_err][pos_err]).sum().item()
            if w != 0:
                G.add_edge(node_id("err", pos_err, l_err), tgt, weight=w)

    # feat sources
    for pos_s, l_s, f_s, a_s in active_feats:
        dec_stack = clt.d[l_s].w[:, l_t - l_s, f_s].detach()
        grads = torch.stack(grad_resid[l_s:l_t])[:, pos_s]
        contrib = (dec_stack * grads).sum().item()
        w = a_s * contrib
        if w != 0:
            G.add_edge(node_id("feat", pos_s, l_s, f_s), tgt, weight=w)

# iterate over logit targets
for tok_out in outs:
    tgt = node_id("logit", tok_out)
    log_scalar = logits[0, -1, tok_out] - logits[0, -1].mean()
    grad_emb0, grad_resid_all = torch.autograd.grad(
        log_scalar, [embed] + resid_nodes, retain_graph=True
    )
    grad_resid_all = [g[0] for g in grad_resid_all]  # TODO: remove

    # emb sources
    for pos_s in range(T):
        tok_id = toks[0, pos_s].item()
        w = (embed[tok_id] * grad_resid_all[0][pos_s]).sum().item()
        if w != 0:
            G.add_edge(node_id("emb", pos_s), tgt, weight=w)

    # err sources
    for l_err in range(L):
        for pos_err in range(T):
            v_out = error[0, pos_err, l_err]
            w = (v_out * grad_resid_all[l_err][pos_err]).sum().item()

    # feat sources
    for pos_s, l_s, f_s, a_s in active_feats:
        dec_stack = clt.d[l_s].w[:, f_s].detach()
        grads = torch.stack(grad_resid_all[l_s:])[:, pos_s]
        contrib = (dec_stack * grads).sum().item()
        w = a_s * contrib
        if w != 0:
            G.add_edge(node_id("feat", pos_s, l_s, f_s), tgt, weight=w)
