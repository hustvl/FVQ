import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from .quantizer import FactorizedVQ
from .feature_decoder import FeatureDecoder


class VisionRotary(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        # Keep CPU FP32 frequencies, unaffected by Module.to(bfloat16).
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, length):
        return torch.outer(torch.arange(length).float(), self.inv_freq)


class VisionEncoder(Qwen3VLVisionModel):
    def __init__(self, config, rotary_mode="native"):
        config = Qwen3VLVisionConfig(**config)
        config._attn_implementation = "sdpa"
        super().__init__(config)
        self.rotary_mode = rotary_mode
        if rotary_mode == "fp32_cpu":
            self.rotary_pos_emb = VisionRotary(config.hidden_size // config.num_heads // 2)
        elif rotary_mode != "native":
            raise ValueError("Unknown vision rotary precision")
        del self.merger
        self.deepstack_merger_list = nn.ModuleList()

    def forward(self, pixels, grid_thw):
        self.config._attn_implementation = ("flash_attention_2" if pixels.is_cuda
            and self.patch_embed.proj.weight.dtype in (torch.float16, torch.bfloat16) else "sdpa")
        grid = grid_thw.cpu() if self.rotary_mode == "fp32_cpu" else grid_thw
        hidden = self.patch_embed(pixels.to(self.patch_embed.proj.weight.dtype))
        hidden = hidden + self.fast_pos_embed_interpolate(grid).to(hidden.device, hidden.dtype)
        rotary = self.rot_pos_emb(grid)
        emb = torch.cat((rotary, rotary), -1)
        position = (emb.cos().to(hidden.device), emb.sin().to(hidden.device))
        lengths = torch.repeat_interleave(grid_thw[:, 1]*grid_thw[:, 2], grid_thw[:, 0])
        cu = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)).to(hidden.device)
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens=cu, position_embeddings=position)
        return hidden


class FVQTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision = VisionEncoder(config["vision"], config.get("vision_rotary", "native"))
        self.vision.requires_grad_(False)
        dim = config["vision"]["hidden_size"]
        self.post_norm = nn.LayerNorm(dim, eps=1e-6)
        self.quantizer = FactorizedVQ(dim=dim, **config["quantizer"])
        self.feature_decoder = (FeatureDecoder(output_dim=dim, **config["decoder"])
                                if config.get("decoder") else None)

    def features(self, pixels, grid_thw):
        with torch.no_grad():
            hidden = self.vision(pixels, grid_thw)
        return self.post_norm(hidden)

    def forward(self, pixels, grid_thw):
        target = self.features(pixels, grid_thw)
        quantized, result = self.quantizer(target)
        if self.feature_decoder is not None:
            reconstructed = self.feature_decoder(quantized, grid_thw)
            result["reconstruction"] = reconstructed
            result["feature_loss"] = F.mse_loss(reconstructed.float(), target.detach().float())
            result["loss"] = result["feature_loss"] + result["quantizer_loss"]
        return result

    @torch.no_grad()
    def encode(self, pixels, grid_thw):
        return self.quantizer(self.features(pixels, grid_thw), update=False)[1]["ids"]

    @classmethod
    def from_pretrained(cls, path, device="cpu", dtype=torch.float32):
        path = Path(path)
        config = json.loads((path / "fvq_config.json").read_text())
        model = cls(config)
        state = load_file(str(path / "fvq.safetensors"))
        model.load_state_dict(state, strict=True)
        return model.to(device=device, dtype=dtype).eval()

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fvq_config.json").write_text(json.dumps(self.config, indent=2))
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(path / "fvq.safetensors"))
