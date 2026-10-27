"""PT byte-chunk sampling and bounded-shuffle packing, without storage clients."""
import json
import math
from pathlib import Path
import random

from cache import fingerprint, greedy_pack, sources, tokenizer_fingerprint
from processing import Overlength


class PretrainStream:
    def __init__(self, manifest, processor, cache_dir, rank=0, world=1, seed=20230324,
                 chunk_bytes=32768, pool_size=1024, overlength="error"):
        if world <= 0 or not 0 <= rank < world or chunk_bytes <= 0 or pool_size <= 0:
            raise ValueError("Invalid rank/world size or chunk/pool size")
        self.processor, self.rank, self.world, self.seed = processor, rank, world, seed
        self.pool_size, self.overlength = pool_size, overlength
        self.files, _ = sources(manifest)
        if any(path.suffix != ".jsonl" for path, _ in self.files):
            raise ValueError("Streaming PT requires JSONL sources")
        settings = {"schema": 1, "rank": rank, "world": world, "seed": seed,
                    "chunk_bytes": chunk_bytes, "pool_size": pool_size, "overlength": overlength,
                    "max_length": processor.max_length, "image": vars(processor.image_processor),
                    "tokenizer": tokenizer_fingerprint(processor.tokenizer),
                    "processing": Path(__file__).with_name("processing.py").read_text()}
        settings["stream_code"] = Path(__file__).read_text()
        self.key = fingerprint(self.files, settings)
        directory = Path(cache_dir)/self.key
        directory.mkdir(parents=True, exist_ok=True)
        index = directory/"chunks.json"
        if index.exists():
            self.chunks = json.loads(index.read_text())
        else:
            self.chunks, past = [], 0
            files = list(self.files)
            random.Random(20230324).shuffle(files)  # nosec B311 -- match initial file ordering
            rng = random.Random(seed+rank)  # nosec B311 -- explicit sampling seed
            for file_index, (path, rate) in enumerate(files, 1):
                size = path.stat().st_size
                count = math.ceil(size/chunk_bytes)
                with path.open("rb") as handle:
                    for chunk in range(count):
                        if (past+chunk) % world != rank:
                            continue
                        handle.seek(min(size, chunk*chunk_bytes))
                        if handle.tell():
                            self.readline(handle)
                        begin = handle.tell()
                        handle.seek(min(size, (chunk+1)*chunk_bytes))
                        self.readline(handle)
                        end = handle.tell()
                        if end > begin:
                            left = rate
                            while left > 0:
                                if left >= rng.random():
                                    self.chunks.append([str(path), begin, end-begin])
                                left -= 1
                        if chunk and chunk % (5000*world) == rank:
                            print(f"PT index rank={rank} file={file_index}/{len(files)} "
                                  f"sampled_chunks={len(self.chunks)}", flush=True)
                past += count
                print(f"PT index rank={rank} file={file_index}/{len(files)} complete "
                      f"sampled_chunks={len(self.chunks)}", flush=True)
            random.Random(rank).shuffle(self.chunks)  # nosec B311 -- initial per-rank chunk shuffle
            temporary = index.with_suffix(".building")
            temporary.write_text(json.dumps(self.chunks))
            temporary.replace(index)
        self.epoch, self.cursor, self.pending, self.filtered = 0, 0, [], 0
        self.order = list(range(len(self.chunks)))
        self.rng = random.Random(seed+rank)  # nosec B311

    @staticmethod
    def readline(handle):
        limit = 64*1024*1024
        line = handle.readline(limit+1)
        if len(line) > limit:
            raise ValueError("JSONL record exceeds 64 MiB")
        return line

    def start_epoch(self, epoch):
        self.epoch, self.cursor, self.pending = epoch, 0, []
        self.order = list(range(len(self.chunks)))
        self.rng = random.Random(self.seed+self.rank+epoch)  # nosec B311
        if epoch:
            self.rng.shuffle(self.order)

    def next_records(self):
        while not self.pending:
            raw = []
            while len(raw) < self.pool_size and self.cursor < len(self.order):
                path, offset, length = self.chunks[self.order[self.cursor]]
                self.cursor += 1
                with open(path, "rb") as handle:
                    handle.seek(offset)
                    end = offset+length
                    while handle.tell() < end:
                        position = handle.tell()
                        line = self.readline(handle)
                        if not line:
                            raise ValueError("JSONL file changed after chunk indexing")
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            row = self.processor.normalize(row, Path(path).parent)
                        except (ValueError, KeyError, TypeError):
                            raise ValueError(f"Invalid pretraining source {Path(path).name} at byte {position}") from None
                        raw.append((row, Path(path).name, position))
                self.rng.shuffle(raw)
            if not raw:
                return None
            records = []
            for row, source, position in raw:
                try:
                    records.append(self.processor.pretrain(row))
                except Overlength:
                    if self.overlength != "filter":
                        raise Overlength(f"Overlength pretraining source {source} at byte {position}") from None
                    self.filtered += 1
                except (ValueError, KeyError, TypeError, OSError):
                    raise ValueError(f"Pretraining preprocessing failed in {source} at byte {position}") from None
            items = ((index, len(record["ids"])) for index, record in enumerate(records))
            self.pending = [[records[index] for index in pack]
                            for pack in greedy_pack(items, self.processor.max_length)]
            self.rng.shuffle(self.pending)
        # Pop before returning: state is the next *consumed* microbatch, not a prefetch pointer.
        return self.pending.pop(0)

    def state_dict(self):
        return {"key": self.key, "epoch": self.epoch, "cursor": self.cursor,
                "pending": list(self.pending), "filtered": self.filtered, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state["key"] != self.key:
            raise ValueError("Data/config/rank/world-size changed; cannot resume the same stream")
        self.start_epoch(state["epoch"])
        self.cursor, self.pending, self.filtered = state["cursor"], state["pending"], state["filtered"]
        rng = state["rng"]
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
