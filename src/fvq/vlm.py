# Attention layout adapted from Hugging Face Transformers Qwen3 (Apache-2.0).
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Modified: FP32 CPU rotary arithmetic and legacy math-SDPA inference.
# See licenses/transformers-LICENSE and THIRD_PARTY_NOTICES.md.
"""The FVQ visual bridge and Qwen3 language model, without legacy audio modules."""
import json
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, repeat_kv
from .tokenizer import FVQTokenizer


class RotaryFP32(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.theta = config.rope_theta
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def forward(self, x, position_ids):
        with torch.autocast(x.device.type, enabled=False):
            freq = position_ids.cpu().float()[..., None] * self.inv_freq
            emb = torch.cat((freq, freq), -1)
            return emb.cos().to(x.device), emb.sin().to(x.device)


class AttentionFP32RoPE(Qwen3Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = (p.unsqueeze(1) for p in position_embeddings)
        def rotate(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), -1)
        q = (q.float()*cos + rotate(q.float())*sin).to(q.dtype)
        k = (k.float()*cos + rotate(k.float())*sin).to(k.dtype)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx,
                           {"sin": sin, "cos": cos, "cache_position": cache_position})
        lengths = kwargs.get("sequence_lengths")
        if lengths is not None:
            if past_key_values is not None or sum(lengths) != q.shape[-2] or q.shape[0] != 1:
                raise ValueError("Packed SFT attention requires one uncached packed sequence")
            if any(length <= 0 for length in lengths):
                raise ValueError("Packed sequence lengths must be positive")
            if self.training and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                # One fused varlen launch, with GQA and no dense L x L mask.
                from flash_attn import flash_attn_varlen_func
                cu = torch.tensor([0]+list(lengths), device=q.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
                qs, ks, vs = (value.transpose(1,2).squeeze(0).contiguous() for value in (q,k,v))
                output = flash_attn_varlen_func(qs, ks, vs, cu, cu, max(lengths), max(lengths), causal=True)
                output = output.unsqueeze(0).transpose(1,2)
            else:
                # Reference/FP32 path: independent causal segments.
                k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
                outputs, start = [], 0
                for length in lengths:
                    stop = start+length
                    context = nullcontext() if self.training else sdpa_kernel(SDPBackend.MATH)
                    with context:
                        outputs.append(F.scaled_dot_product_attention(q[:, :, start:stop],
                            k[:, :, start:stop], v[:, :, start:stop], is_causal=True))
                    start = stop
                output = torch.cat(outputs, dim=2)
        else:
            k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
            # Match legacy inference's math SDPA. Training may select fused SDPA.
            context = nullcontext() if self.training else sdpa_kernel(SDPBackend.MATH)
            with context:
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                            is_causal=attention_mask is None and q.shape[-2] > 1)
        output = output.transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1)
        return self.o_proj(output), None


class FVQVLM(nn.Module):
    def __init__(self, tokenizer, llm, config):
        super().__init__()
        self.tokenizer, self.llm, self.config = tokenizer, llm, config
        self.tokenizer.requires_grad_(False).eval()
        dim = tokenizer.config["vision"]["hidden_size"]
        groups, size = tokenizer.quantizer.groups, tokenizer.quantizer.size
        self.embeddings = nn.ModuleList([nn.Embedding(size+1, dim//groups) for _ in range(groups)])
        self.merger = nn.Module()
        self.merger.ln_q = nn.LayerNorm(dim, eps=1e-6)
        self.merger.mlp = nn.Sequential(nn.Linear(dim*4, dim*4), nn.GELU(),
                                       nn.Linear(dim*4, llm.config.hidden_size))
        self.llm.model.rotary_emb = RotaryFP32(llm.config)
        for layer in self.llm.model.layers:
            layer.self_attn.__class__ = AttentionFP32RoPE

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def visual_embeddings(self, pixel_values, grid_thw):
        with torch.no_grad():
            ids = self.tokenizer.encode(pixel_values, grid_thw)
        features = torch.cat([embedding(ids[:, g]) for g, embedding in enumerate(self.embeddings)], -1)
        return self.merger.mlp(self.merger.ln_q(features).reshape(-1, features.shape[-1]*4))

    def inputs_embeds(self, input_ids, pixel_values=None, grid_thw=None):
        x = self.llm.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            visual = self.visual_embeddings(pixel_values, grid_thw)
            mask = input_ids == self.config["image_pad_token_id"]
            if mask.sum().item() != len(visual):
                raise ValueError("Image marker count does not match visual tokens")
            x = x.clone()
            x[mask] = visual.to(x.dtype)
        return x

    def forward(self, input_ids, pixel_values=None, grid_thw=None, labels=None, **kwargs):
        if labels is not None:
            ignored = self.config.get("loss_ignore_token_ids", [self.config["image_pad_token_id"]])
            labels = labels.clone()
            for token in ignored:
                labels[input_ids == token] = -100
        dtype = self.llm.get_input_embeddings().weight.dtype
        # ZeRO BF16 disables outer AMP; enable it inside the module so the
        # tokenizer's FP32 inputs can use its BF16 projection parameters.
        amp = (torch.autocast(input_ids.device.type, dtype=torch.bfloat16)
               if dtype == torch.bfloat16 else nullcontext())
        with amp:
            x = self.inputs_embeds(input_ids, pixel_values, grid_thw)
            return self.llm(inputs_embeds=x, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, grid_thw=None, max_new_tokens=128, **kwargs):
        dtype = self.llm.get_input_embeddings().weight.dtype
        amp = (torch.autocast(input_ids.device.type, dtype=torch.bfloat16)
               if dtype == torch.bfloat16 else nullcontext())
        with amp:
            x = self.inputs_embeds(input_ids, pixel_values, grid_thw)
            return self.llm.generate(input_ids=input_ids, inputs_embeds=x,
                                     max_new_tokens=max_new_tokens, **kwargs)

    @classmethod
    def from_pretrained(cls, path, device="cpu", dtype=torch.float32):
        from safetensors.torch import load_file
        path = Path(path)
        config = json.loads((path / "vlm_config.json").read_text())
        if config.get("interleaved_mrope", False):
            raise ValueError("This minimal release targets the verified 1D RoPE checkpoint")
        tokenizer = FVQTokenizer.from_pretrained(path, device=device, dtype=dtype)
        llm = Qwen3ForCausalLM.from_pretrained(path / "llm", dtype=dtype,
                           attn_implementation="sdpa", device_map={"": device},
                           local_files_only=True)  # nosec B615 -- local converted weights only
        llm.config.use_cache = True
        model = cls(tokenizer, llm, config)
        model.embeddings.to(device=device, dtype=dtype)
        model.merger.to(device=device, dtype=dtype)
        state = load_file(str(path / "bridge.safetensors"), device=str(device))
        model.embeddings.load_state_dict({k.removeprefix("embeddings."): v
                                         for k,v in state.items() if k.startswith("embeddings.")}, strict=True)
        model.merger.load_state_dict({k.removeprefix("merger."): v
                                     for k,v in state.items() if k.startswith("merger.")}, strict=True)
        return model.eval()

    def save_pretrained(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        self.tokenizer.save_pretrained(path)
        source_name = self.llm.config._name_or_path
        self.llm.config._name_or_path = ""
        try:
            self.llm.save_pretrained(path / "llm")
        finally:
            self.llm.config._name_or_path = source_name
        state = {k: v.detach().cpu().contiguous() for k,v in self.state_dict().items()
                 if k.startswith(("embeddings.", "merger."))}
        save_file(state, str(path / "bridge.safetensors"))
        (path / "vlm_config.json").write_text(json.dumps(self.config, indent=2))
