"""Export trusted legacy weights to a standalone FVQ directory.

This tool copies model weights only; it never copies legacy Python/config files.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config, AutoTokenizer
from fvq import FVQTokenizer


def vision_config(config):
    keys = ("depth", "hidden_size", "intermediate_size", "num_heads", "in_channels",
            "patch_size", "temporal_patch_size", "spatial_merge_size", "hidden_act",
            "out_hidden_size", "num_position_embeddings")
    result = {k: config[k] for k in keys if k in config}
    result["deepstack_visual_indexes"] = []
    return result


def tokenizer_config(vision, decoder):
    return {"vision": vision_config(vision), "vision_rotary": "native" if decoder else "fp32_cpu", "quantizer": {
        "groups": 16, "group_dim": 144, "size": 16384,
        "output_dim": 1024 if decoder else None, "decay": 0.99,
        "commitment_weight": 0.25, "restart": True, "seed": 0},
        "decoder": {"dim": 1024, "depth": 3, "heads": 16, "base_grid": 16} if decoder else None}


def map_tokenizer_key(key):
    for prefix in ("encoder.frozen_blocks.", "encoder.trainable_blocks."):
        if key.startswith(prefix):
            return "vision.blocks." + key.removeprefix(prefix)
    for source, target in (("encoder.qwen_vit.", "vision."),
                           ("encoder.qwen_post_layernorm.", "post_norm."),
                           ("quantize.", "quantizer.")):
        if key.startswith(source):
            return target + key.removeprefix(source)
    if key.startswith("feature_decoder."):
        return key
    raise ValueError(f"Unrecognized tokenizer tensor: {key}")


def export_tokenizer(args):
    config = json.loads((args.vision / "config.json").read_text())
    config = config.get("vision_config", config)
    model = FVQTokenizer(tokenizer_config(config, decoder=True))
    if args.source.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(args.source))
    else:
        if not args.trust_pickle:
            raise ValueError("Only convert user-owned trusted pickle files with --trust-pickle")
        state = torch.load(args.source, map_location="cpu", weights_only=True)
    mapped = {map_tokenizer_key(k): v for k,v in state.items()
              if not k.startswith("quantize.codebook_used")}
    model.load_state_dict(mapped, strict=True)
    model.save_pretrained(args.output)


def export_vlm(args):
    source = args.source
    config = json.loads((source / "config.json").read_text())
    visual, quant = config["visual_config"], config["visual_quantizer_config"]
    required = {"quantizer_type": "fvq", "vq_at_vit_output": True,
                "quant_conv_linear_only": True, "pq_embed_cat": True, "pq_embed_proj": False,
                "depth": 16, "group_dim": 144, "codebook_size": 16384}
    for key, value in required.items():
        if quant.get(key) != value:
            raise ValueError(f"Unsupported checkpoint setting: {key}")
    if visual.get("use_interleaved_mrope", False) or visual.get("use_res_mod_zjl", False):
        raise ValueError("This converter targets the verified plain-RoPE, no-ResMod checkpoint")
    tok = FVQTokenizer(tokenizer_config(visual, decoder=False))
    tok_state, bridge = {}, {}
    llm_path = args.output / "llm"
    llm_path.mkdir()
    weight_map, size = {}, 0
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    skipped = []
    for shard_id, shard in enumerate(shards):
        llm_state = {}
        with safe_open(source / shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                v = f.get_tensor(k)
                if k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head.")):
                    llm_state[k] = v
                elif k.startswith("model.visual_model."):
                    if ".deepstack_merger_list." in k:
                        skipped.append(k)
                    else:
                        tok_state[k.replace("model.visual_model.", "vision.")] = v
                elif k.startswith("model.visual_quantizer.pre_norm."):
                    tok_state[k.replace("model.visual_quantizer.pre_norm.", "post_norm.")] = v
                elif k.startswith("model.visual_quantizer.quant_conv."):
                    tok_state[k.replace("model.visual_quantizer.quant_conv.", "quantizer.in_proj.")] = v
                elif k.startswith("model.visual_quantizer.quantize."):
                    nk = k.replace("model.visual_quantizer.quantize.", "quantizer.")
                    if "codebook_used" in nk:
                        skipped.append(k)
                    else:
                        if nk.endswith(".embed"):
                            nk = nk.removesuffix(".embed") + ".embedding.weight"
                            v = v[:-1]  # Only the excluded lookup padding row.
                        tok_state[nk] = v.float().contiguous()
                elif k.startswith("model.visual_bridge_model.embedding_layers."):
                    bridge[k.replace("model.visual_bridge_model.embedding_layers.", "embeddings.")] = v
                elif k.startswith("model.visual_input_bridge."):
                    nk = k.replace("model.visual_input_bridge.", "merger.")
                    nk = nk.replace(".norm.", ".ln_q.").replace(".linear_fc1.", ".mlp.0.").replace(".linear_fc2.", ".mlp.2.")
                    bridge[nk] = v
                else:
                    raise ValueError(f"Unrecognized VLM tensor: {k}")
        if llm_state:
            filename = f"model-{shard_id+1:05d}.safetensors"
            save_file(llm_state, str(llm_path / filename), metadata={"format": "pt"})
            weight_map.update({k:filename for k in llm_state})
            size += sum(v.numel()*v.element_size() for v in llm_state.values())
    # Legacy VLM checkpoints did not contain tokenizer usage EMA, and are not
    # suitable for resuming tokenizer training. Fill only that unused inference buffer.
    for g in range(tok.quantizer.groups):
        key = f"quantizer.codebooks.{g}.ema_usage"
        tok_state[key] = tok.state_dict()[key]
    tok.load_state_dict(tok_state, strict=True)
    tok.save_pretrained(args.output)
    save_file(bridge, str(args.output / "bridge.safetensors"))
    llm_keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "max_position_embeddings", "rope_theta",
                "rms_norm_eps", "vocab_size", "hidden_act", "tie_word_embeddings", "eos_token_id", "pad_token_id")
    llm_config = Qwen3Config(**{k:config[k] for k in llm_keys if k in config}, rope_scaling=None)
    llm_config.save_pretrained(llm_path)
    (llm_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map}))
    public_config = {"image_pad_token_id": visual["image_pad_token_id"], "interleaved_mrope": False,
                     "loss_ignore_token_ids": config["multimodal_special_token_no_loss_list"],
                     "min_pixels": visual["min_pixels"], "max_pixels": visual["max_pixels"]}
    (args.output / "vlm_config.json").write_text(json.dumps(public_config, indent=2))
    AutoTokenizer.from_pretrained(source, trust_remote_code=False,
        local_files_only=True).save_pretrained(args.output / "text_tokenizer")  # nosec B615 -- local source assets
    print(f"Excluded {len(skipped)} inactive deepstack/diagnostic tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["tokenizer", "vlm"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision", type=Path, help="Public Qwen3-VL vision config directory")
    parser.add_argument("--trust-pickle", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing export directory")
    if args.kind == "tokenizer" and args.vision is None:
        parser.error("--vision is required for tokenizer conversion")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    (export_tokenizer if args.kind == "tokenizer" else export_vlm)(args)


if __name__ == "__main__":
    main()
