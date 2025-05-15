import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import einsum, reduce, rearrange
from math import sqrt


def rectangle(x: Tensor) -> Tensor:
    return ((x > -0.5) & (x < 0.5)).type_as(x)


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x * (x > threshold)).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = grad_output.clone()
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class DecoderLayer(nn.Module):
    def __init__(self, idx: int, n_layers: int, n_features: int, d_model: int):
        super().__init__()
        self.w = nn.Parameter(
            torch.empty(n_layers - idx, n_features // n_layers, d_model)
        )
        nn.init.uniform_(
            self.w.data,
            -1 / sqrt(n_layers * d_model),
            1 / sqrt(n_layers * d_model),
        )

    def forward(self, x: Tensor):
        out = einsum(self.w, x, "l f d, t f -> t l d")
        return out


class CLT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_features: int,
        bandwidth: float,
        threshold: float,
        lambda_p: float,
        c: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.bandwidth = bandwidth
        self.lambda_p = lambda_p
        self.c = c

        self.w_e = nn.Parameter(torch.empty(n_layers, d_model, n_features // n_layers))
        nn.init.uniform_(self.w_e.data, -1 / sqrt(n_features), 1 / sqrt(n_features))

        self.d = nn.ModuleList(
            [DecoderLayer(i, n_layers, n_features, d_model) for i in range(n_layers)]
        )

        self.t = nn.Parameter(torch.full((n_layers, n_features // n_layers), threshold))

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = einsum(self.w_e, x, "l d f, t l d -> t l f")
        acts: Tensor = JumpReLU.apply(h, torch.exp(self.t), self.bandwidth)  # type: ignore
        return h, acts

    def decode(self, acts: Tensor) -> Tensor:
        T, L, _ = acts.shape
        x_hat = acts.new_zeros(T, L, self.d_model)

        for i, dec in enumerate(self.d):
            x_hat[:, i:] += dec(acts[:, i])

        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h, acts = self.encode(x)
        x_hat = self.decode(acts)
        return h, acts, x_hat

    def get_loss(
        self, x: Tensor, h: Tensor, x_hat: Tensor, acts: Tensor, lambda_s: float
    ) -> Tensor:
        l_mse = F.mse_loss(x_hat, x)
        l_p = self.lambda_p * F.relu(-h)
        l_p = reduce(l_p, "t l f -> t l", "sum").mean()

        l_s = 0
        for i, dec in enumerate(self.d):
            w_flat = rearrange(dec.w, "l f d -> f (l d)")
            w_norm = torch.norm(w_flat, dim=-1)

            l_s += torch.tanh(self.c * w_norm * acts[:, i]).sum()

        return l_mse + l_p + lambda_s * l_s
