import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from einops import einsum, reduce, rearrange
from math import sqrt


def rectangle(x: Tensor) -> Tensor:
    return ((x > -0.5) & (x < 0.5)).type_as(x)


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float):
        threshold = threshold.to(x.dtype)
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x * (x > threshold)).type_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class LocalDecoder(nn.Module):
    def __init__(self, idx: int, n_layers: int, f_local: int, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_layers - idx, f_local, d_model))
        nn.init.uniform_(
            self.w.data,
            -1 / sqrt(n_layers * d_model),
            1 / sqrt(n_layers * d_model),
        )

    def forward(self, x: Tensor):
        out = einsum(self.w, x, "l f d, t f -> t l d")
        return out


class InferenceCLT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_features: int,
        bandwidth: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.bandwidth = bandwidth
        self.f_layer = n_features // n_layers
        self.w_e = nn.Parameter(torch.empty(n_layers, d_model, self.f_layer))

        self.d = nn.ModuleList(
            [LocalDecoder(i, n_layers, self.f_layer, d_model) for i in range(n_layers)]
        )
        self.t = nn.Parameter(torch.empty((n_layers, self.f_layer)))

    @classmethod
    def from_pretrained(cls, sd):
        model = cls(sd["n_layers"], sd["d_model"], sd["n_features"], sd["bandwidth"])
        model.load_state_dict(sd["model_state"])
        model.eval()
        return model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        T, L, _ = x.shape
        h = einsum(self.w_e, x, "l d f, t l d -> t l f")
        acts: Tensor = JumpReLU.apply(h, torch.exp(self.t), self.bandwidth)  # type: ignore
        x_hat = acts.new_zeros(T, L, self.d_model)

        for i, dec in enumerate(self.d):
            x_hat[:, i:] += dec(acts[:, i])
        return h, acts, x_hat


class FeatureParallelCLT(nn.Module):
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
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.lambda_p = lambda_p
        self.c = c

        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

        f_layer = n_features // n_layers
        f_local = f_layer // self.world
        self.f_local = f_local

        self.w_e = nn.Parameter(torch.empty(n_layers, d_model, f_local))
        nn.init.uniform_(self.w_e.data, -1 / sqrt(n_features), 1 / sqrt(n_features))

        self.d = nn.ModuleList(
            [LocalDecoder(i, n_layers, f_local, d_model) for i in range(n_layers)]
        )

        self.t = nn.Parameter(torch.full((n_layers, f_local), threshold))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        T, L, _ = x.shape

        h = einsum(self.w_e, x, "l d f, t l d -> t l f")
        acts: Tensor = JumpReLU.apply(h, torch.exp(self.t), self.bandwidth)  # type: ignore

        x_hat = acts.new_zeros(T, L, self.d_model)

        for i, dec in enumerate(self.d):
            x_hat[:, i:] += dec(acts[:, i])

        x_hat = x_hat.contiguous()

        return h, acts, x_hat

    @torch.compile
    def get_loss(
        self, x: Tensor, h: Tensor, x_hat: Tensor, acts: Tensor, lambda_s: float
    ) -> Tensor:
        l_mse = F.mse_loss(x_hat, x)
        l_p = self.lambda_p * F.relu(-h)
        l_p = reduce(l_p, "t l f -> t l", "sum").mean()

        get_norm = lambda dec: torch.norm(rearrange(dec.w, "l f d -> f (l d)"), dim=-1)
        w_norm = torch.stack([get_norm(dec) for dec in self.d], dim=0)
        l_s = torch.tanh(self.c * w_norm * acts).sum()

        return l_mse + l_p + lambda_s * l_s
