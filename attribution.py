import torch
from torch import Tensor
from einops import rearrange
from transformer_lens import HookedTransformer, utils
import networkx as nx
import numpy as np

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


def _compute_normalized_adjacency_matrix(G: nx.DiGraph, nodes: list) -> np.ndarray:
    A = np.abs(nx.to_numpy_array(G, nodelist=nodes, weight="weight")).T

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    A = A / row_sums

    return A


def _get_logit_weights(nodes: list, logits: Tensor) -> np.ndarray:
    probs = torch.softmax(logits[0, -1], dim=-1)
    weights = np.zeros(len(nodes))

    for i, node in enumerate(nodes):
        if node[0] == "logit":
            tok = node[1]
            weights[i] = probs[tok].item()

    return weights


def prune_graph(
    G: nx.DiGraph,
    logits: Tensor,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    max_logit_nodes: int = 10,
    logit_prob_threshold: float = 0.95,
) -> nx.DiGraph:
    nodes = list(G.nodes())

    # Step 1: Prune logit nodes separately
    logit_nodes = [n for n in nodes if n[0] == "logit"]
    if logit_nodes:
        probs = torch.softmax(logits[0, -1], dim=-1)
        logit_probs = [(n, probs[n[1]].item()) for n in logit_nodes]
        logit_probs.sort(key=lambda x: x[1], reverse=True)

        cumsum = 0.0
        kept_logits = []
        for node, prob in logit_probs:
            if cumsum >= logit_prob_threshold or len(kept_logits) >= max_logit_nodes:
                break
            kept_logits.append(node)
            cumsum += prob
    else:
        kept_logits = []

    # Step 2: Prune non-logit nodes by indirect influence
    A = _compute_normalized_adjacency_matrix(G, nodes)

    # Calculate indirect influence matrix: B = (I - A)^-1 - I
    I = np.eye(A.shape[0])
    B = np.linalg.inv(I - A) - I

    # Get logit weights
    logit_weights = _get_logit_weights(nodes, logits)

    # Calculate influence on logits for each node
    influence_on_logits = B @ logit_weights

    # Sort non-logit nodes by influence
    node_influences = [
        (nodes[i], influence_on_logits[i])
        for i in range(len(nodes))
        if nodes[i][0] != "logit"
    ]
    node_influences.sort(key=lambda x: x[1], reverse=True)

    # Keep nodes with cumulative influence up to threshold
    total_influence = sum(inf for _, inf in node_influences)
    if total_influence > 0:
        cumsum = 0.0
        kept_nodes = []
        for node, inf in node_influences:
            if cumsum / total_influence >= node_threshold:
                break
            kept_nodes.append(node)
            cumsum += inf
    else:
        kept_nodes = [n for n, _ in node_influences]

    # Always keep embedding and error nodes
    kept_nodes.extend([n for n in nodes if n[0] in ("emb", "err")])
    kept_nodes.extend(kept_logits)

    # Create subgraph with kept nodes
    G_pruned = G.subgraph(kept_nodes).copy()

    # Step 3: Prune edges by thresholded influence
    nodes_pruned = list(G_pruned.nodes())
    A_pruned = _compute_normalized_adjacency_matrix(G_pruned, nodes_pruned)

    # Recalculate influence matrix
    I_pruned = np.eye(A_pruned.shape[0])
    B_pruned = np.linalg.inv(I_pruned - A_pruned) - I_pruned

    # Get node scores (influence on logits)
    logit_weights_pruned = _get_logit_weights(nodes_pruned, logits)
    node_scores = B_pruned @ logit_weights_pruned

    # For logit nodes, use their probability as score
    for i, node in enumerate(nodes_pruned):
        if logit_weights_pruned[i] > 0:
            node_scores[i] = logit_weights_pruned[i]

    # Calculate edge scores (edge_score = A * node_score[:, None])
    # This creates a matrix where edge_score[i,j] = A[i,j] * node_score[i]
    edge_score_matrix = A_pruned * node_scores[:, np.newaxis]

    # Convert to list of (edge, score) tuples
    edge_scores = []
    for u, v, data in G_pruned.edges(data=True):
        u_idx = nodes_pruned.index(u)
        v_idx = nodes_pruned.index(v)
        score = edge_score_matrix[v_idx, u_idx]  # Remember A is transposed
        edge_scores.append(((u, v), score))

    # Sort edges by score and calculate cumulative threshold
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    total_edge_score = sum(score for _, score in edge_scores)

    if total_edge_score > 0:
        cumsum = 0.0
        edges_to_keep = []
        for (u, v), score in edge_scores:
            cumsum += score
            edges_to_keep.append((u, v))
            if cumsum / total_edge_score >= edge_threshold:
                break

        # Remove edges not in keep list
        all_edges = list(G_pruned.edges())
        edges_to_remove = [e for e in all_edges if e not in edges_to_keep]
        G_pruned.remove_edges_from(edges_to_remove)

    return G_pruned


def graph_stats(G: nx.DiGraph) -> dict:
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "node_types": {},
        "edge_types": {},
    }

    # node types
    for node in G.nodes():
        node_type = node[0]
        stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1

    # edge types
    for u, v in G.edges():
        edge_type = f"{u[0]}->{v[0]}"
        stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1

    return stats
