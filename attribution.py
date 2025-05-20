import torch
from torch import Tensor
from einops import rearrange
from transformer_lens import HookedTransformer, utils

from clt import InferenceCLT

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The National Digital Analytics Group (N"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=dtype)  # type: ignore

L = model.cfg.n_layers

sd = torch.load("model.pt", map_location=device)
clt: InferenceCLT = InferenceCLT.from_pretrained(sd).to(device)

with torch.no_grad():
    toks = model.to_tokens(PROMPT)
    logits_ref, cache_ref = model.run_with_cache(  # type: ignore
        toks,
        names_filter=lambda n: (
            n.endswith("hook_pattern")
            or n.endswith("hook_scale")
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


for l in range(L):  # noqa: E741
    freeze(model, "pattern", l)
    freeze(model, "scale", l, "ln1")
    freeze(model, "scale", l, "ln2")

pre = torch.stack([cache_ref["normalized", i, "ln2"] for i in range(clt.n_layers)])
pre = rearrange(pre, "l b t d -> (b t) l d")
with torch.no_grad():
    _, acts_full, x_hat_full = clt(pre)
acts = rearrange(acts_full, "(b t) l f -> b t l f", b=toks.size(0))
x_hat = rearrange(x_hat_full, "(b t) l d -> b t l d", b=toks.size(0))
error = cache_ref.stack("mlp_out") - x_hat

for i in range(L):
    mlp_out = (x_hat[:, :, i] + error[:, :, i]).detach()

    def _mlp_replacer(tensor, hook):
        return mlp_out

    model.add_hook(utils.get_act_name("mlp_out", i), _mlp_replacer)

embed = model.W_E
embed.requires_grad_(True)

logits, cache = model.run_with_cache(
    toks,
    names_filter=lambda n: n.endswith("normalized") or n == "embed",
    return_type=None,
)

resid_nodes = []
for l in range(L):  # noqa: E741
    cache["normalized", l, "ln2"].requires_grad_(True)
    resid_nodes.append(cache["normalized", l, "ln2"])


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


def w_emb_to_feat(tok_idx: int, l_t: int, f_t: int, pos_t: int):
    v_in = clt.w_e[l_t, :, f_t].detach()
    grad_embed, _ = _backward_to_sources(l_t, pos_t, v_in)
    v_out = embed[:, tok_idx].detach()
    return (v_out * grad_embed[:, tok_idx]).sum()


def w_emb_to_out(tok_src: int, tok_out: int, pos_out: int):
    logit_scalar = logits[0, pos_out, tok_out] - logits[0, pos_out].mean()
    grad_embed = torch.autograd.grad(logit_scalar, embed, retain_graph=True)[0]
    v_out = embed[:, tok_src].detach()
    return (v_out * grad_embed[:, tok_src]).sum()


def w_feat_to_feat(pos_s: int, l_s: int, f_s: int, pos_t: int, l_t: int, f_t: int):
    v_in = clt.w_e[l_t, :, f_t].detach()
    _, grad_resid = _backward_to_sources(l_t, pos_t, v_in)
    total = 0.0
    for l in range(l_s, l_t):  # noqa: E741
        decoder_vec = clt.d[l_s].w[l - l_s, f_s].detach()  # type: ignore
        grad_slice = grad_resid[l]
        total += (decoder_vec * grad_slice[:, pos_s]).sum()
    a_s = acts[0, pos_s, l_s, f_s].detach()
    return a_s * total


def w_feat_to_out(pos_s: int, l_s: int, f_s: int, tok_out: int, pos_out: int):
    logit_scalar = logits[0, pos_out, tok_out] - logits[0, pos_out].mean()
    grad_full = torch.autograd.grad(
        logit_scalar, resid_nodes, retain_graph=True, allow_unused=True
    )
    total = 0.0
    for l in range(l_s, L):  # noqa: E741
        decoder_vec = clt.d[l_s].w[l_s, f_s].detach()
        grad_slice = grad_full[l]
        total += (decoder_vec * grad_slice[:, pos_s]).sum()
    a_s = acts[0, pos_s, l_s, f_s].detach()
    return a_s * total


def w_err_to_feat(pos_err: int, l_err: int, pos_t: int, l_t: int, f_t: int):
    v_out = error[:, pos_err, l_err].detach()
    v_in = clt.w_e[l_t, :, f_t].detach()
    _, grad_resid = _backward_to_sources(l_t, pos_t, v_in)
    grad_slice = grad_resid[l_err]
    return (v_out * grad_slice[:, pos_err]).sum()


def w_err_to_out(pos_err: int, l_err: int, tok_out: int, pos_out: int):
    logit_scalar = logits[0, pos_out, tok_out] - logits[0, pos_out].mean()
    grad_err = torch.autograd.grad(logit_scalar, resid_nodes[l_err], retain_graph=True)[
        0
    ]
    grad_slice = grad_err[:, pos_err]
    v_out = error[:, pos_err, l_err].detach()
    return (v_out * grad_slice).sum()
