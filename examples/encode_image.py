import argparse
import torch
from fvq import FVQTokenizer, ImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--image', required=True)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
model = FVQTokenizer.from_pretrained(args.checkpoint, args.device,
                                    torch.bfloat16 if args.device.startswith('cuda') else torch.float32)
batch = {k:v.to(args.device) for k,v in ImageProcessor()([args.image]).items()}
with torch.autocast(args.device.split(':')[0], dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
    ids = model.encode(batch['pixel_values'], batch['grid_thw'])
print('ID shape:', tuple(ids.shape))
print(ids[:4].tolist())
