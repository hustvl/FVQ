import math
from pathlib import Path

import torch


def training_paths(parser, args, config):
    """CLI path overrides take priority over the stage YAML."""
    for name in ("checkpoint", "data", "output", "cache"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name) or config.get(name)
        if name == "cache" and value is None:
            value = "cache"
        if not isinstance(value, str) or not value.strip():
            parser.error(f"Set {name} in the YAML or pass --{name}")
        setattr(args, name, value)


class ModelEMA:
    """EMA parameters; persistent buffers are copied live, never averaged/reset."""
    def __init__(self, model, decay=0.9999):
        self.decay, self.step = decay, 0
        self.params = {k: p.detach().clone() for k,p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        n = self.step-1
        decay = min(self.decay, (1+n)/(10+n)) if n > 0 else 0.0
        for key, p in model.named_parameters():
            if p.requires_grad:
                self.params[key].sub_((1-decay)*(self.params[key]-p))
            else:
                self.params[key].copy_(p)

    def state_dict(self):
        return {"params": self.params, "step": self.step, "decay": self.decay}

    def load_state_dict(self, state):
        if state["params"].keys() != self.params.keys():
            raise ValueError("EMA checkpoint parameter structure differs")
        self.params = {k: v.to(device=self.params[k].device, dtype=self.params[k].dtype)
                       for k,v in state["params"].items()}
        self.step, self.decay = state["step"], state["decay"]

    def save_pretrained(self, model, path):
        from safetensors.torch import save_file
        import json
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
        for k,v in self.params.items():
            state[k] = v.detach().cpu().contiguous()
        save_file(state, str(path/"fvq.safetensors"))
        (path/"fvq_config.json").write_text(json.dumps(model.config, indent=2))


def scheduler(optimizer, config):
    warmup, steps = config["warmup_steps"], config["steps"]
    minimum = config.get("min_lr", config["lr"])/config["lr"]
    def multiplier(step):
        if step < warmup:
            return step/max(1, warmup)
        if config.get("scheduler", "cosine") == "constant":
            return 1.0
        progress = min(1., (step-warmup)/max(1, steps-warmup))
        return minimum+(1-minimum)*0.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
