"""Training formats are deliberately separate from the inference chat template."""
import re
import json
from pathlib import Path

import torch
from PIL import Image


class Overlength(ValueError):
    pass


def split_marked(text, pattern):
    parts, end = [], 0
    for match in pattern.finditer(text):
        if match.start() > end:
            parts.append((text[end:match.start()], False))
        parts.append((match.group(), True))
        end = match.end()
    if end < len(text):
        parts.append((text[end:], False))
    return parts


class TrainingProcessor:
    ROLE_TOKENS = {"system": "<B_SYS>", "user_system": "<B_USYS>",
                   "user": "<C_Q>", "assistant": "<C_A>", "function": "<B_FUNC>",
                   "code": "<B_CODE>", "ape": "<B_APE>"}
    SEPARATORS = ("<function_calling>", "<calc_start>", "<calc_end>", "<inner_think>")
    IMAGE = re.compile(r"<img_start>.*?<img_end>", re.S)
    TRAINABLE = re.compile(r"<trainable_start>.*?<trainable_end>", re.S)

    def __init__(self, tokenizer, image_processor, max_length):
        self.tokenizer, self.image_processor, self.max_length = tokenizer, image_processor, max_length
        vocab = tokenizer.get_vocab()
        required = list(self.ROLE_TOKENS.values()) + list(self.SEPARATORS)
        required += ["<img_start>", "<img_end>", "<img_pad>"]
        if any(token not in vocab for token in required):
            raise ValueError("Training requires the exported role/image token vocabulary")
        self.roles = {role: vocab[token] for role, token in self.ROLE_TOKENS.items()}
        self.separators = {token: vocab[token] for token in self.SEPARATORS}
        self.separators["<assistant_continue>"] = self.roles["assistant"]
        self.separator_pattern = re.compile("|".join(map(re.escape, self.separators)))
        self.image_ids = [vocab[token] for token in ("<img_start>", "<img_end>", "<img_pad>")]

    def normalize(self, row, base):
        """Normalize public/local media; retain explicit learnable and literal think text."""
        base = Path(base)
        def media(path):
            if isinstance(path, str) and path.lstrip().startswith("{"):
                info = json.loads(path)
                if info.get("type") == "s3":
                    raise ValueError("Materialize object-store images as local/mounted files before training")
                path = info.get("local", info.get("path"))
            if not isinstance(path, str) or "://" in path:
                raise ValueError("Only local or mounted image files are supported")
            return str((base / path).resolve())
        def content(value):
            if isinstance(value, list):
                pieces = []
                for item in value:
                    if item.get("type") == "text":
                        pieces.append(item["text"])
                    elif item.get("type") == "image":
                        pieces.append("<img_start>"+media(item["text"])+"<img_end>")
                    else:
                        raise ValueError("Unsupported multimodal content type")
                value = "".join(pieces)
            if not isinstance(value, str):
                raise ValueError("Content must be a string or typed text/image list")
            return self.IMAGE.sub(lambda m: "<img_start>"+media(m.group()[11:-9])+"<img_end>", value)
        if isinstance(row, list):
            if len(row) < 2:
                raise ValueError("List-format pretraining record requires its text at index 1")
            return {"content": content(row[1])}
        result = dict(row)
        messages = row.get("messages", row.get("conversations"))
        if messages is not None:
            result["messages"] = []
            for message in messages:
                role = message.get("role", message.get("from"))
                role = {"human": "user", "gpt": "assistant"}.get(role, role)
                out = {"role": role, "content": content(message.get("content", message.get("value")))}
                if "learnable" in message:
                    if not isinstance(message["learnable"], bool):
                        raise ValueError("learnable must be boolean")
                    out["learnable"] = message["learnable"]
                result["messages"].append(out)
        for key in ("input", "content"):
            if key in result:
                result[key] = content(result[key])
        images = row.get("images", [row["image"]] if "image" in row else [])
        images = iter(media(path) for path in images)
        def replace(value):
            while "<image>" in value:
                try:
                    path = next(images)
                except StopIteration as error:
                    raise ValueError("Missing image for <image> marker") from error
                value = value.replace("<image>", "<img_start>"+path+"<img_end>", 1)
            return value
        for key in ("input", "content"):
            if key in result:
                result[key] = replace(result[key])
        for message in result.get("messages", []):
            message["content"] = replace(message["content"])
        if next(images, None) is not None:
            raise ValueError("Unused image paths; add explicit image markers")
        return result

    def text(self, text):
        if any(tag in text for tag in ("<audio_start>", "<video_start>", "<videoframe_start_omni>")):
            raise ValueError("Audio/video training is outside this release")
        parts = split_marked(text, self.TRAINABLE)
        if not parts:
            raise ValueError("Empty training content")
        single = len(parts) == 1
        ids, labels, images, grids = [], [], [], []
        for part, trainable in parts:
            part = part.replace("<trainable_start>", "").replace("<trainable_end>", "")
            if not single and not part.strip():
                continue
            trainable = single or trainable
            for piece, is_image in split_marked(part, self.IMAGE):
                if is_image:
                    path = piece[11:-9]
                    with Image.open(path) as image:
                        if image.width*image.height <= 16**2:
                            raise ValueError("Image is too small for the original training format")
                        h, w = self.image_processor.resized_shape(*reversed(image.size))
                    p = self.image_processor.patch_size
                    grids.append([1, h//p, w//p])
                    images.append(path)
                    start, end, pad = (self.tokenizer.convert_ids_to_tokens(i) for i in self.image_ids)
                    piece = start+pad*(h*w//(p*self.image_processor.merge_size)**2)+end
                tokens = self.tokenizer.encode(piece, add_special_tokens=False)
                ids.extend(tokens)
                labels.extend(tokens if trainable else [-100]*len(tokens))
        if not ids:
            raise ValueError("Training record has no tokens")
        return {"ids": ids, "labels": labels, "images": images, "grids": grids}

    def pretrain(self, row):
        title = row.get("title", "")
        if "input" in row or "content" in row:
            value = row["input"] if "input" in row else row["content"]
            text = title+value
        else:
            pieces = []
            for message in row.get("messages", []):
                role, value = message["role"], message["content"]
                if role in ("user", "system"):
                    pieces.append(value)
                elif role == "assistant":
                    pieces.append("<trainable_start>"+value+"<trainable_end>")
                else:
                    raise ValueError("Unsupported pretraining message role")
            text = title+"".join(pieces)
        result = self.text(text)
        eos = self.tokenizer.eos_token_id
        result["ids"].append(eos)
        result["labels"].append(eos if row.get("eosloss", True) else -100)
        if len(result["ids"]) > self.max_length:
            raise Overlength("Pretraining record exceeds max_length")
        return result

    def sft(self, row):
        ids, labels, images, grids, learnable = [], [], [], [], False
        messages = row.get("messages")
        if not messages:
            raise ValueError("SFT requires nonempty messages/conversations")
        previous_role = None
        for index, message in enumerate(messages):
            role = message["role"]
            if role not in self.roles or role == previous_role:
                raise ValueError("Unsupported or repeated SFT role")
            if role == "system" and index != 0 or role == "user_system" and index > 1:
                raise ValueError("System messages must precede conversation turns")
            previous_role = role
            role = "system" if role == "user_system" else role
            role_id = self.roles[role]
            ids.append(role_id)
            labels.append((self.tokenizer.eos_token_id if role == "user" else role_id) if learnable else -100)
            if not message["content"].strip():
                raise ValueError("Empty SFT message")
            mm = self.text(message["content"])
            raw = self.tokenizer.decode(mm["ids"], skip_special_tokens=False)
            # Preserve the original decode/retokenize boundary and separator mapping.
            tokens = []
            for piece, is_separator in split_marked(raw, self.separator_pattern):
                tokens.extend([self.separators[piece]] if is_separator else
                              self.tokenizer.encode(piece, add_special_tokens=False))
            ids.extend(tokens)
            learnable = message.get("learnable", role == "assistant")
            labels.extend(tokens if learnable else [-100]*len(tokens))
            images.extend(mm["images"]); grids.extend(mm["grids"])
            if len(ids) >= self.max_length:
                raise Overlength("SFT record reaches/exceeds max_length before terminal EOS")
        if learnable:
            ids.append(self.tokenizer.eos_token_id); labels.append(self.tokenizer.eos_token_id)
        if not any(label >= 0 for label in labels):
            raise ValueError("SFT record contains no supervised targets")
        return {"ids": ids, "labels": labels, "images": images, "grids": grids}

    def batch(self, records, mode, loss_ignore_token_ids=()):
        ids, labels, positions, lengths, images, grids = [], [], [], [], [], []
        for record in records:
            ids.extend(record["ids"]); labels.extend(record["labels"])
            length = len(record["ids"])
            positions.extend(range(length)); lengths.append(length)
            images.extend(record["images"]); grids.extend(record["grids"])
        real_length = len(ids)
        if not real_length or real_length > self.max_length:
            raise ValueError("Invalid packed sequence length")
        ignored = set(loss_ignore_token_ids) | set(self.image_ids)
        if mode == "sft":
            ignored.update(self.roles.values()); ignored.update(self.separators.values())
        labels = [-100 if token in ignored else label for token, label in zip(ids, labels)]
        if not any(label >= 0 for label in labels[1:]):
            raise ValueError("Packed sequence has no effective supervised targets")
        padding = self.max_length-real_length
        ids.extend([0 if mode == "sft" else self.tokenizer.pad_token_id]*padding)
        labels.extend([-100]*padding); positions.extend([0]*padding)
        batch = {"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels])}
        if mode == "sft":
            batch["position_ids"] = torch.tensor([positions])
            batch["sequence_lengths"] = tuple(lengths+([padding] if padding else []))
        # PT intentionally uses continuous positions/causal attention, as its trainer did.
        if images:
            batch.update(self.image_processor(images))
            if not torch.equal(batch["grid_thw"], torch.tensor(grids)):
                raise ValueError("Image dimensions changed after cache construction")
        return batch
