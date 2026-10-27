"""Tight-frame FVQ with independent EMA codebooks (feature reconstruction)."""
import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
            return (y * self.weight).to(x.dtype)


def projection(in_dim, out_dim):
    return nn.Sequential(RMSNorm(in_dim), nn.Linear(in_dim, in_dim),
                         nn.GELU(), nn.Linear(in_dim, out_dim))


class Codebook(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("embed_ema", self.embedding.weight.detach().clone())
        self.register_buffer("cluster_size_ema", torch.zeros(size))
        self.register_buffer("ema_usage", torch.full((size,), 1.0 / size))

    def _apply(self, fn, recurse=True):
        # Device movement is allowed; rounding the saved codebook is not.
        values = {"embedding": self.embedding.weight.detach().clone(),
                  **{k: v.clone() for k, v in self._buffers.items()}}
        super()._apply(fn, recurse)
        device = self.embedding.weight.device
        self.embedding.weight.data = values.pop("embedding").to(device=device)
        for key, value in values.items():
            self._buffers[key] = value.to(device=device)
        return self


class FactorizedVQ(nn.Module):
    def __init__(self, dim=1152, groups=16, group_dim=144, size=16384,
                 output_dim=1024, decay=0.99, commitment_weight=0.25,
                 restart=True, seed=0):
        super().__init__()
        if groups * group_dim < dim:
            raise ValueError("The frame needs groups * group_dim >= dim")
        self.dim, self.groups, self.size = dim, groups, size
        self.rho = groups * group_dim / dim
        self.decay, self.commitment_weight, self.restart = decay, commitment_weight, restart
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = projection(dim, output_dim) if output_dim else nn.Identity()
        matrix = torch.randn(groups * group_dim, dim, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(seed))
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        self.register_buffer("R", (math.sqrt(self.rho) * q).float().reshape(groups, group_dim, dim))
        self.codebooks = nn.ModuleList([Codebook(size, group_dim) for _ in range(groups)])

    def _apply(self, fn, recurse=True):
        frame = self.R.clone()
        super()._apply(fn, recurse)
        self.R = frame.to(device=self.R.device)
        return self

    @staticmethod
    def nearest(x, codebook):
        # Operation ordering matters at ties and for checkpoint ID alignment.
        x, codebook = x.float(), codebook.float()
        distance = torch.addmm(x.pow(2).sum(1, keepdim=True)
                               + codebook.t().pow(2).sum(0, keepdim=True),
                               x, codebook.t(), alpha=-2.0)
        return distance.argmin(-1)

    @torch.no_grad()
    def _reset(self, cb, x):
        dead = torch.nonzero(cb.ema_usage < 0.03 / self.size).flatten()
        if not dead.numel():
            return
        reset = dead[torch.randperm(len(dead), device=x.device)]
        if len(x) < self.size:
            x = x.repeat(math.ceil(self.size / len(x)), 1)
            x = x + torch.rand_like(x) * (torch.ones(x.shape[-1], device=x.device)
                                         * (0.01 / math.sqrt(x.shape[-1])))
        values = x[torch.randperm(len(x), device=x.device)[:len(reset)]].float()
        if dist.is_initialized():
            dist.broadcast(reset, 0)
            dist.broadcast(values, 0)
        cb.embedding.weight[reset] = values
        cb.embed_ema[reset] = values
        cb.cluster_size_ema[reset] = 1
        cb.ema_usage[reset] = 1.0 / self.size

    def forward(self, features, update=None):
        update = self.training if update is None else update
        latent = self.in_proj(features.float())
        analyzed = torch.einsum("...d,grd->...gr", latent, self.R)
        chunks = [analyzed[..., g, :].reshape(-1, analyzed.shape[-1]) for g in range(self.groups)]
        ids = [self.nearest(x, cb.embedding.weight) for x, cb in zip(chunks, self.codebooks)]
        statistics, pending = [], []
        if update:
            with torch.no_grad():
                for x, index in zip(chunks, ids):
                    one_hot = torch.zeros(self.size, len(x), device=x.device)
                    one_hot.scatter_(0, index[None], 1)
                    counts = one_hot.sum(1)
                    sums = (one_hot @ x.float()).float()
                    usage = counts / len(x)
                    if dist.is_initialized():
                        for value in (sums, counts, usage):
                            pending.append(dist.all_reduce(value, async_op=True))
                    statistics.append((counts, sums, usage))
                for request in pending:
                    request.wait()
                for cb, (counts, sums, usage) in zip(self.codebooks, statistics):
                    if dist.is_initialized():
                        usage /= dist.get_world_size()
                    cb.cluster_size_ema.mul_(self.decay).add_(counts, alpha=1-self.decay)
                    cb.embed_ema.mul_(self.decay).add_(sums, alpha=1-self.decay)
                    n = cb.cluster_size_ema.sum()
                    norm = n * (cb.cluster_size_ema + 1e-5) / (n + self.size * 1e-5)
                    cb.embedding.weight.copy_(cb.embed_ema / norm[:, None])
                    cb.ema_usage.mul_(self.decay).add_(usage, alpha=1-self.decay)
        vectors = [F.embedding(i, cb.embedding.weight).to(x.dtype)
                   for i, x, cb in zip(ids, chunks, self.codebooks)]
        quant = torch.einsum("...gr,grd->...d", torch.stack(vectors, -2), self.R) / self.rho
        quant = quant.reshape_as(latent)
        commitment = self.commitment_weight * (latent - quant.detach()).pow(2).mean()
        vq_loss = (quant - latent.detach()).pow(2).mean()
        if update and self.restart:
            for cb, x in zip(self.codebooks, chunks):
                self._reset(cb, x.detach().float())
        straight_through = latent + (quant - latent).detach()
        return self.out_proj(straight_through), {
            "ids": torch.stack(ids, -1).reshape(*latent.shape[:-1], self.groups),
            "commitment_loss": commitment, "vq_loss": vq_loss,
            "quantizer_loss": commitment + vq_loss,
        }
