"""Local PT streaming or cached SFT; frozen discrete tokenizer in every phase."""
import argparse
import json
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoTokenizer
from fvq import FVQVLM, ImageProcessor
from processing import TrainingProcessor
from cache import SFTCache
from stream import PretrainStream
from common import scheduler, training_paths


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "data", "output", "cache"):
        parser.add_argument("--"+name, help=f"Override {name} in the stage YAML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="A training snapshot, not a cross-stage model initialization")
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    training_paths(parser, args, cfg)
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(mixed_precision=cfg["precision"], gradient_accumulation_steps=cfg["accumulation"],
                              step_scheduler_with_optimizer=False,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None:
        plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        plugin.deepspeed_config["gradient_accumulation_steps"] = cfg["accumulation"]
    mode = cfg["data_mode"]
    if mode not in ("pretrain", "sft"):
        raise ValueError("data_mode must be pretrain or sft")
    if cfg.get("overlength", "error") not in ("error", "filter"):
        raise ValueError("overlength must be error or filter")
    torch.manual_seed(cfg["seed"])
    checkpoint = Path(args.resume or args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint/"text_tokenizer", local_files_only=True,
                                              trust_remote_code=False)  # nosec B615 -- local assets
    processor = TrainingProcessor(tokenizer, ImageProcessor(max_pixels=cfg["max_pixels"]), cfg["max_length"])
    if mode == "sft":
        data = SFTCache(args.data, processor, args.cache, cfg["seed"], cfg.get("overlength", "error"),
                        cfg.get("disable_packing", False))
    else:
        data = PretrainStream(args.data, processor, args.cache, accelerator.process_index,
                              accelerator.num_processes, cfg["seed"], overlength=cfg.get("overlength", "error"))
    state, step, first_epoch = None, 0, 0
    if args.resume:
        state = json.loads((checkpoint/f"data-rank-{accelerator.process_index}.json").read_text())
        if state["config"] != cfg:
            raise ValueError("Resume requires the same training config; omit --resume for a new stage")
        step, first_epoch = state["step"], state["data"]["epoch"]

    def start_epoch(epoch):
        if mode == "sft":
            if accelerator.is_main_process:
                data.start_epoch(epoch)
            accelerator.wait_for_everyone()
            data.start_epoch(epoch, accelerator.process_index, accelerator.num_processes)
        else:
            data.start_epoch(epoch)

    start_epoch(first_epoch)
    if state:
        data.load_state_dict(state["data"])
    # Cache failures are detected before allocating the 8B model.
    model = FVQVLM.from_pretrained(checkpoint, dtype=torch.float32)
    if cfg["precision"] == "bf16":
        model.tokenizer.vision.to(torch.bfloat16)
    model.llm.requires_grad_(cfg["train_llm"])
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.config.use_cache = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                 weight_decay=cfg["weight_decay"], foreach=False)
    schedule = scheduler(optimizer, cfg)
    model, optimizer, schedule = accelerator.prepare(model, optimizer, schedule)
    if args.resume:
        accelerator.load_state(str(checkpoint/"training_state"))
    ignored = accelerator.unwrap_model(model).config.get("loss_ignore_token_ids", ())

    def save():
        out = Path(args.output)/f"step-{step}"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(out)
            tokenizer.save_pretrained(out/"text_tokenizer")
        accelerator.wait_for_everyone()
        accelerator.save_state(str(out/"training_state"))
        (out/f"data-rank-{accelerator.process_index}.json").write_text(json.dumps(
            {"step": step, "config": cfg, "data": data.state_dict()}))
        accelerator.wait_for_everyone()

    model.train()
    saved_step = None
    for epoch in range(first_epoch, cfg["epochs"]):
        if epoch != first_epoch:
            start_epoch(epoch)
        while step < cfg["steps"]:
            before = data.state_dict()
            group = [data.next_records() for _ in range(cfg["accumulation"])]
            ready = torch.tensor(int(all(records is not None for records in group)), device=accelerator.device)
            if accelerator.num_processes > 1:
                torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            if not ready.item():
                data.load_state_dict(before)
                accelerator.print("Epoch exhausted on at least one rank; unmatched/incomplete accumulation tail dropped")
                break
            for records in group:
                batch = processor.batch(records, mode, ignored)
                batch = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                with accelerator.accumulate(model), accelerator.autocast():
                    result = model(**batch, use_cache=False)
                    accelerator.backward(result.loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(parameters, cfg["clip_grad"])
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        schedule.step()
                    optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                accelerator.print(f"epoch={epoch} step={step} loss={result.loss.item():.6f} "
                                  f"overlength_filtered={getattr(data, 'filtered', 0)}")
                if step % cfg["save_steps"] == 0 or step == cfg["steps"]:
                    save(); saved_step = step
        if step >= cfg["steps"]:
            break
    if step and saved_step != step:
        save()


if __name__ == "__main__":
    main()
