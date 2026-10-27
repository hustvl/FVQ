import torch
from torch import nn
from torch.nn import functional as F
from .quantizer import projection


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, lengths):
        qkv = self.qkv(x).view(len(x), 3, self.heads, -1)
        if x.is_cuda:
            from flash_attn import flash_attn_varlen_func
            cu = F.pad(torch.tensor(lengths, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
            y = flash_attn_varlen_func(qkv[:, 0], qkv[:, 1], qkv[:, 2], cu, cu,
                                      max(lengths), max(lengths))
        else:
            parts = []
            for chunk in qkv.split(lengths):
                q, k, v = [chunk[:, i].transpose(0, 1).unsqueeze(0) for i in range(3)]
                parts.append(F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1))
            y = torch.cat(parts)
        return self.proj(y.reshape(len(x), -1))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn, self.mlp = Attention(dim, heads), MLP(dim)

    def forward(self, x, lengths):
        x = x + self.attn(self.norm1(x), lengths)
        return x + self.mlp(self.norm2(x))


class FeatureDecoder(nn.Module):
    def __init__(self, dim=1024, output_dim=1152, depth=3, heads=16, base_grid=16):
        super().__init__()
        self.base_grid = base_grid
        self.positional_embedding = nn.Parameter(torch.randn(base_grid**2, dim) * dim**-0.5)
        self.ln_pre, self.ln_post = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.transformer = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.decode_task_layer = projection(dim, output_dim)

    def forward(self, x, grid_thw):
        grids = grid_thw.tolist()
        lengths = [h * w for t, h, w in grids]
        if any(t != 1 for t, _, _ in grids) or sum(lengths) != len(x):
            raise ValueError("Feature decoding supports packed still images only")
        parts = []
        for chunk, (_, h, w) in zip(x.split(lengths), grids):
            pos = self.positional_embedding
            if (h, w) != (self.base_grid, self.base_grid):
                pos = pos.reshape(self.base_grid, self.base_grid, -1).permute(2, 0, 1)[None]
                pos = F.interpolate(pos, (h, w), mode="bicubic", align_corners=False)
                pos = pos[0].permute(1, 2, 0).reshape(h*w, -1)
            # Preserve the trained positional order; do not raster-reorder patches.
            parts.append(chunk + pos.to(x.dtype))
        x = self.ln_pre(torch.cat(parts))
        for block in self.transformer:
            x = block(x, lengths)
        return self.decode_task_layer(self.ln_post(x))
