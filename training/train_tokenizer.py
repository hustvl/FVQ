"""Feature-only tokenizer training. Launch with accelerate launch or Python."""
import argparse
import torch
import yaml
from accelerate import Accelerator
from fvq import FVQTokenizer, ImageProcessor
from data import JSONLData
from common import ModelEMA, scheduler, training_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Override checkpoint in the stage YAML")
    parser.add_argument("--data", help="Override data in the stage YAML")
    parser.add_argument("--config", default="configs/tokenizer_g16.yaml")
    parser.add_argument("--output", help="Override output in the stage YAML")
    parser.add_argument("--resume", help="Accelerate training-state directory, not model weights")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    training_paths(parser, args, cfg)
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(mixed_precision=cfg["precision"],
                               gradient_accumulation_steps=cfg["accumulation"],
                               step_scheduler_with_optimizer=False)
    torch.manual_seed(cfg["seed"])
    model = FVQTokenizer.from_pretrained(args.checkpoint)
    if model.feature_decoder is None:
        raise ValueError("A full tokenizer checkpoint with feature decoder is required")
    # Match the trained frozen-backbone BF16 compute; trainable parameters remain FP32.
    if cfg["precision"] == "bf16" and accelerator.device.type == "cuda":
        model.vision.to(torch.bfloat16)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    schedule = scheduler(optimizer, cfg)
    model, optimizer, schedule = accelerator.prepare(model, optimizer, schedule)
    bare = accelerator.unwrap_model(model)
    ema = ModelEMA(bare)
    accelerator.register_for_checkpointing(ema)
    if args.resume:
        accelerator.load_state(args.resume)
    step = ema.step
    data = JSONLData(args.data, accelerator.process_index, accelerator.num_processes, cfg["seed"])
    processor = ImageProcessor(min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"])
    model.train()
    while step < cfg["steps"]:
        count = 0
        for row in data:
            count += 1
            batch = {k: v.to(accelerator.device) for k,v in processor(row["images"]).items()}
            with accelerator.accumulate(model), accelerator.autocast():
                result = model(batch["pixel_values"], batch["grid_thw"])
                accelerator.backward(result["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, cfg["clip_grad"])
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    schedule.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                step += 1
                ema.update(bare)
                accelerator.print(f"step={step} loss={result['loss'].item():.6f}")
                if step % cfg["save_steps"] == 0 or step == cfg["steps"]:
                    out = f"{args.output}/step-{step}"
                    accelerator.save_state(out+"/training_state")
                    if accelerator.is_main_process:
                        bare.save_pretrained(out+"/model")
                        ema.save_pretrained(bare, out+"/ema")
                if step >= cfg["steps"]:
                    break
        if count == 0:
            raise ValueError("No training records on this rank")


if __name__ == "__main__":
    main()
