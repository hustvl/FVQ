"""Small public JSONL interface: local images and chat messages, no remote services."""
import json
import random
from pathlib import Path
from torch.utils.data import IterableDataset


class JSONLData(IterableDataset):
    def __init__(self, path, rank=0, world=1, seed=42, shuffle_buffer=1024):
        self.path = Path(path)
        self.rank, self.world, self.seed, self.buffer = rank, world, seed, shuffle_buffer
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)  # nosec B311 -- data shuffling, not cryptography
        self.epoch += 1
        buffer = []
        with self.path.open() as f:
            for index, line in enumerate(f):
                if index % self.world != self.rank:
                    continue
                try:
                    row = json.loads(line)
                    images = row.get("images", [row["image"]] if "image" in row else [])
                    row["images"] = [str(self.path.parent / image) for image in images]
                except Exception as error:
                    raise ValueError(f"Invalid JSONL record at line {index+1}") from error
                buffer.append(row)
                if len(buffer) >= self.buffer:
                    yield buffer.pop(rng.randrange(len(buffer)))
        rng.shuffle(buffer)
        yield from buffer
