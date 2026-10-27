"""Initialize FVQ from local public Qwen3-VL vision weights for feature training."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from fvq import FVQTokenizer
from convert_checkpoint import tokenizer_config

parser = argparse.ArgumentParser()
parser.add_argument('--vision', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError('Refusing to overwrite an existing checkpoint')
torch.set_num_threads(4)
config = json.loads((args.vision/'config.json').read_text())
model = FVQTokenizer(tokenizer_config(config['vision_config'], decoder=True))
state = model.state_dict()
index = json.loads((args.vision/'model.safetensors.index.json').read_text())['weight_map']
needed = {k:v for k,v in index.items() if k.startswith('model.visual.')}
loaded = set()
for shard in sorted(set(needed.values())):
    with safe_open(args.vision/shard, framework='pt', device='cpu') as f:
        for k in f.keys():
            nk = k.replace('model.visual.', 'vision.')
            if nk in state:
                state[nk] = f.get_tensor(k)
                loaded.add(nk)
            elif k in ('model.visual.merger.norm.weight','model.visual.merger.norm.bias'):
                nk = k.replace('model.visual.merger.norm.','post_norm.')
                state[nk] = f.get_tensor(k)
                loaded.add(nk)
expected = {k for k in state if k.startswith(('vision.','post_norm.'))}
if loaded != expected:
    raise ValueError(f'Incomplete vision checkpoint: {sorted(expected-loaded)}')
model.load_state_dict(state,strict=True)
model.save_pretrained(args.output)
