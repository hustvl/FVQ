import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling


class ImageProcessor:
    def __init__(self, min_pixels=50176, max_pixels=4194304, patch_size=16,
                 merge_size=2, temporal_patch_size=2):
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.patch_size, self.merge_size = patch_size, merge_size
        self.temporal_patch_size = temporal_patch_size

    def resized_shape(self, h, w):
        if max(h, w) / min(h, w) > 200:
            raise ValueError("Image aspect ratio exceeds 200")
        factor = self.patch_size*self.merge_size
        rh, rw = max(factor, round(h/factor)*factor), max(factor, round(w/factor)*factor)
        if rh*rw > self.max_pixels:
            beta = math.sqrt(h*w/self.max_pixels)
            rh, rw = math.floor(h/beta/factor)*factor, math.floor(w/beta/factor)*factor
        elif rh*rw < self.min_pixels:
            beta = math.sqrt(self.min_pixels/(h*w))
            rh, rw = math.ceil(h*beta/factor)*factor, math.ceil(w*beta/factor)*factor
        if min(rh, rw) < factor:
            raise ValueError("Pixel budget is too small for this aspect ratio")
        return rh, rw

    def __call__(self, images):
        pixels, grids = [], []
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        for image in images:
            image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
            w, h = image.size
            rh, rw = self.resized_shape(h, w)
            array = resize(np.array(image), (rh, rw), PILImageResampling.BICUBIC).transpose(2, 0, 1)
            array = (array / 255.0 - np.array([0.5]*3)[:, None, None]) / np.array([0.5]*3)[:, None, None]
            gh, gw = rh//p, rw//p
            array = np.tile(array[None], (t, 1, 1, 1)).reshape(1, t, 3, gh//m, m, p, gw//m, m, p)
            array = array.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh*gw, 3*t*p*p)
            pixels.append(torch.from_numpy(array).float())
            grids.append([1, gh, gw])
        return {"pixel_values": torch.cat(pixels), "grid_thw": torch.tensor(grids)}


class ChatProcessor:
    """Images correspond to <image> markers, in conversation order."""
    def __init__(self, tokenizer_path, image_processor=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False,
                          local_files_only=True)  # nosec B615 -- local exported assets only
        self.image_processor = image_processor or ImageProcessor()

    def __call__(self, messages, images=(), add_generation_prompt=True, enable_thinking=True):
        batch = self.image_processor(images) if images else {}
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                  add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        if text.count("<image>") != len(images):
            raise ValueError("Exactly one <image> marker is needed for each image")
        for _, h, w in batch.get("grid_thw", []):
            start, end, pad = [self.tokenizer.convert_ids_to_tokens(i) for i in (151672, 151673, 151674)]
            marker = start + pad*(int(h*w)//4) + end
            text = text.replace("<image>", marker, 1)
        batch["input_ids"] = self.tokenizer(text, return_tensors="pt").input_ids
        return batch
