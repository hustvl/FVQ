import argparse
import torch
from fvq import FVQVLM, ChatProcessor, ImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--image', required=True)
parser.add_argument('--prompt', default='Describe this image.')
parser.add_argument('--max-new-tokens', type=int, default=128)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
device = args.device
model = FVQVLM.from_pretrained(args.checkpoint, device, torch.bfloat16)
processor = ChatProcessor(args.checkpoint+'/text_tokenizer', ImageProcessor(
                          min_pixels=model.config['min_pixels'], max_pixels=model.config['max_pixels']))
batch = processor([{'role':'user','content':'<image>\n'+args.prompt}], [args.image])
batch = {k:v.to(device) for k,v in batch.items()}
stop = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids('<|im_end|>')]
with torch.autocast(device.split(':')[0], dtype=torch.bfloat16, enabled=device.startswith('cuda')):
    output = model.generate(**batch, do_sample=False, max_new_tokens=args.max_new_tokens, eos_token_id=stop)
print(processor.tokenizer.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True))
