import torch
from torch import Tensor
from einops import rearrange
from transformer_lens import HookedTransformer, utils
import networkx as nx

from clt import InferenceCLT


def _freeze(model, cache_ref, *name):
    def _hook(tensor, hook):
        return cache_ref[name].detach()

    model.add_hook(utils.get_act_name(name), _hook)  # type: ignore


def _node_id(kind, *args):
    return (kind,) + tuple(args)


def _backward_to_sources(l_t: int, pos_t: int, v_in: Tensor, resid_nodes: list[Tensor]):
    grad_out = torch.zeros_like(resid_nodes[l_t])
    grad_out[pos_t] = v_in

    return torch.autograd.grad(
        outputs=resid_nodes[l_t],
        inputs=resid_nodes,
        grad_outputs=grad_out,
        retain_graph=True,
        allow_unused=True,
    )


def _collect_active_feats(n_toks: int, n_layers: int, acts: Tensor) -> list:
    active = []
    for pos in range(n_toks):
        for layer in range(n_layers):
            nz = (acts[pos, layer] > 0).nonzero(as_tuple=True)[0]
            for f in nz.tolist():
                a_s = acts[pos, layer, f].item()
                active.append((pos, layer, f, a_s))
    return active


def get_vals(model: HookedTransformer, prompt: str, clt: InferenceCLT):
    L = model.cfg.n_layers

    with torch.no_grad():
        toks = model.to_tokens(prompt)[0]
        _, cache_ref = model.run_with_cache(  # type: ignore
            toks,
            names_filter=lambda n: (
                n.endswith("pattern")
                or n.endswith("scale")
                or n.endswith("normalized")
                or n.endswith("mlp_out")
                or n == "embed"
            ),
        )

    for i in range(L):
        _freeze(model, cache_ref, "pattern", i)
        _freeze(model, cache_ref, "scale", i, "ln1")
        _freeze(model, cache_ref, "scale", i, "ln2")

    pre = torch.stack([cache_ref["normalized", i, "ln2"] for i in range(L)])
    post = torch.stack([cache_ref["mlp_out", i] for i in range(L)])
    pre = rearrange(pre, "l b t d -> (b t) l d")
    post = rearrange(post, "l b t d -> (b t) l d")

    with torch.no_grad():
        _, acts, x_hat = clt(pre)

    error = post - x_hat

    for i in range(L):
        mlp_out = (x_hat[:, i] + error[:, i]).detach()

        def _mlp_replacer(tensor, hook):
            return mlp_out

        model.add_hook(utils.get_act_name("mlp_out", i), _mlp_replacer)

    logits, cache = model.run_with_cache(
        toks,
        names_filter=lambda n: n.endswith("normalized") or n == "embed",
        return_type=None,
    )

    resid_nodes = [
        cache["normalized", i, "ln2"].squeeze(0).requires_grad_(True) for i in range(L)
    ]

    return toks, logits.logits, acts, error, resid_nodes


def build_graph(
    toks: Tensor,
    logits: Tensor,
    model: HookedTransformer,
    clt: InferenceCLT,
    acts: Tensor,
    error: Tensor,
    resid_nodes: list[Tensor],
) -> nx.DiGraph:
    G = nx.DiGraph()
    T = toks.size(-1)
    L = model.cfg.n_layers

    embed = model.W_E

    # get top <=10 out toks that account for 95% probability mass
    outs_prob = torch.softmax(logits[0, -1], dim=-1)  # type: ignore
    vals, idxs = outs_prob.topk(10)
    idx = torch.searchsorted(vals.cumsum(-1), 0.95)
    idx = min(idx + 1, vals.size(-1))  # type: ignore
    outs = idxs[:idx].tolist()

    active_feats = _collect_active_feats(T, L, acts)

    # iterate over feature targets
    for pos_t, l_t, f_t, _ in active_feats:
        tgt = _node_id("feat", pos_t, l_t, f_t)
        v_in = clt.w_e[l_t, :, f_t].detach()
        grad_resid = _backward_to_sources(l_t, pos_t, v_in, resid_nodes)

        # emb sources
        for pos_s in range(T):
            v_out = embed[toks[pos_s]]
            w = (v_out * grad_resid[0][pos_s]).sum().item()
            if w != 0:
                G.add_edge(_node_id("emb", pos_s), tgt, weight=w)

        # err sources
        for l_err in range(l_t):
            for pos_err in range(T):
                v_out = error[pos_err, l_err]
                w = (v_out * grad_resid[l_err][pos_err]).sum().item()
                if w != 0:
                    G.add_edge(_node_id("err", pos_err, l_err), tgt, weight=w)

        # feat sources
        for pos_s, l_s, f_s, a_s in active_feats:
            if l_s >= l_t:
                continue
            dec_stack = clt.d[l_s].w[: l_t - l_s, f_s].detach()  # type: ignore
            grads = torch.stack(grad_resid[l_s:l_t])[:, pos_s]
            w = a_s * (dec_stack * grads).sum().item()
            if w != 0:
                G.add_edge(_node_id("feat", pos_s, l_s, f_s), tgt, weight=w)

    # iterate over logit targets
    for tok_out in outs:
        tgt = _node_id("logit", tok_out)
        log_scalar = logits[0, -1, tok_out] - logits[0, -1].mean()
        grad_resid = torch.autograd.grad(log_scalar, resid_nodes, retain_graph=True)

        # emb sources
        for pos_s in range(T):
            v_out = embed[toks[pos_s]]
            w = (v_out * grad_resid[0][pos_s]).sum().item()
            if w != 0:
                G.add_edge(_node_id("emb", pos_s), tgt, weight=w)

        # err sources
        for l_err in range(L):
            for pos_err in range(T):
                v_out = error[pos_err, l_err]
                w = (v_out * grad_resid[l_err][pos_err]).sum().item()
                if w != 0:
                    G.add_edge(_node_id("err", pos_err, l_err), tgt, weight=w)

        # feat sources
        for pos_s, l_s, f_s, a_s in active_feats:
            dec_stack = clt.d[l_s].w[:, f_s].detach()  # type: ignore
            grad = torch.stack(grad_resid[l_s:])[:, pos_s]
            w = a_s * (dec_stack * grad).sum().item()
            if w != 0:
                G.add_edge(_node_id("feat", pos_s, l_s, f_s), tgt, weight=w)

    return G
