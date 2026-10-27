"""Portable manifests and persistent, world-size-independent SFT packing cache."""
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile

import json5

from processing import Overlength


def sources(path):
    path = Path(path).resolve()
    if path.suffix == ".jsonl":
        return [(path, 1.0)], False
    with path.open() as handle:
        config = json5.load(handle)
    if isinstance(config, list):
        return [(path, 1.0)], False
    train = config["train_files"]
    root = (path.parent / train.get("root", "")).resolve()
    result, loaded = [], set()
    for pattern, rate in train["sample_rate"].items():
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
            raise ValueError("Sample rates must be finite and nonnegative")
        candidate = root / pattern
        files = [str(candidate)] if candidate.is_file() else sorted(glob.glob(str(candidate)))
        if not files:
            files = sorted(glob.glob(str(candidate)+"*.json*"))
        if not files:
            raise FileNotFoundError("Manifest pattern matched no files")
        for file in files:
            file = Path(file).resolve()
            if file.suffix not in (".jsonl", ".json", ".json5"):
                raise ValueError("Manifest sources must be JSONL/JSON/JSON5 files")
            if file not in loaded:
                loaded.add(file); result.append((file, float(rate)))
    return result, train.get("stream_sample", False)


def fingerprint(files, settings):
    stamps = [(str(path), rate, path.stat().st_size, path.stat().st_mtime_ns) for path, rate in files]
    return hashlib.sha256(json.dumps([stamps, settings], sort_keys=True).encode()).hexdigest()


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Fast-tokenizer calls mutate these runtime switches; our processor explicitly
    # disables both and applies its own record-length policy.
    backend.pop("truncation", None)
    backend.pop("padding", None)
    return json.dumps(backend, sort_keys=True)


def rows(path):
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                if text.strip():
                    try:
                        yield line, json.loads(text)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON in source {path.name}, line {line}") from None
    else:
        with path.open(encoding="utf-8") as handle:
            data = json5.load(handle) if path.suffix == ".json5" else json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON data source must contain an array")
        yield from enumerate(data, 1)


def greedy_pack(items, max_length, disable_packing=False):
    pack, size = [], 0
    for index, length in items:
        if length > max_length:
            raise ValueError("Cache record exceeds packing limit")
        if pack and (disable_packing or size+length > max_length):
            yield pack
            pack, size = [], 0
        pack.append(index); size += length
    if pack:
        yield pack


def epoch_indices(groups, epoch, seed):
    indices = []
    for start, end, rate in groups:
        count = end-start
        whole = int(rate)
        indices.extend(list(range(start, end))*whole)
        sample = int((rate-whole)*count)
        if sample:
            offset = (epoch*sample) % count
            indices.extend(start+(offset+i) % count for i in range(sample))
    random.Random(seed+epoch+1).shuffle(indices)  # nosec B311 -- reproducible data ordering
    return indices


class SFTCache:
    def __init__(self, manifest, processor, cache_dir, seed=42, overlength="error", disable_packing=False):
        self.processor, self.seed, self.disable_packing = processor, seed, disable_packing
        self.files, self.stream_sample = sources(manifest)
        settings = {"schema": 1, "seed": seed, "overlength": overlength,
                    "stream_sample": self.stream_sample, "packing": not disable_packing,
                    "max_length": processor.max_length,
                    "image": vars(processor.image_processor),
                    "tokenizer": tokenizer_fingerprint(processor.tokenizer),
                    "processing": hashlib.sha256(Path(__file__).with_name("processing.py").read_bytes()).hexdigest()}
        settings["cache_code"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.key = fingerprint(self.files, settings)
        self.directory = Path(cache_dir)/self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/"sft.sqlite"
        self.overlength = overlength
        self.epoch, self.cursor, self.world, self.rank = 0, 0, 1, 0

    def build(self):
        if self.path.exists():
            with sqlite3.connect(self.path) as db:
                if db.execute("select value from metadata where key='complete'").fetchone() != (self.key,):
                    raise ValueError("Incomplete or incompatible SFT cache")
            return
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".building", delete=False) as handle:
            temporary = Path(handle.name)
        statistics = {"input": 0, "selected": 0, "overlength": 0, "duplicates": 0}
        try:
            with sqlite3.connect(temporary) as db:
                db.executescript("""
                    create table raw (id integer primary key, line integer, payload text);
                    create table records (id integer primary key, digest text unique, payload text, length integer);
                    create table groups (start integer, stop integer, rate real);
                    create table plans (epoch integer, id integer, indices text, primary key(epoch,id));
                    create table epochs (epoch integer primary key, count integer);
                    create table metadata (key text primary key, value text);
                """)
                rng = random.Random(self.seed)  # nosec B311
                count = 0
                for source, rate in self.files:
                    db.execute("delete from raw")
                    selected, start = [], count
                    stream_rng = random.Random(f"{self.seed}:{source}")  # nosec B311
                    for line, row in rows(source):
                        statistics["input"] += 1
                        try:
                            row = self.processor.normalize(row, source.parent)
                        except (ValueError, KeyError, TypeError):
                            raise ValueError(f"Invalid SFT format in {source.name}, line {line}") from None
                        if self.stream_sample and rate < 1 and stream_rng.random() >= rate:
                            continue
                        index = len(selected)
                        db.execute("insert into raw values (?,?,?)", (index, line, json.dumps(row)))
                        selected.append(index)
                    if rate < 1 and not self.stream_sample:
                        rng.shuffle(selected)
                        selected = selected[:int(len(selected)*rate)]
                    statistics["selected"] += len(selected)
                    for index in selected:
                        line, payload = db.execute("select line,payload from raw where id=?", (index,)).fetchone()
                        row = json.loads(payload)
                        try:
                            encoded = self.processor.sft(row)
                        except Overlength:
                            if self.overlength != "filter":
                                raise Overlength(f"Overlength SFT record in {source.name}, line {line}") from None
                            statistics["overlength"] += 1
                            continue
                        except (ValueError, KeyError, TypeError, OSError):
                            raise ValueError(f"SFT preprocessing failed in {source.name}, line {line}") from None
                        # Content and explicit learnable are semantic; unrelated metadata is not.
                        digest = hashlib.sha256(json.dumps([row["messages"], len(encoded["ids"])], sort_keys=True).encode()).hexdigest()
                        result = db.execute("insert or ignore into records values (?,?,?,?)",
                                            (count, digest, json.dumps(encoded), len(encoded["ids"])))
                        if result.rowcount:
                            count += 1
                        else:
                            statistics["duplicates"] += 1
                        if count and count % 1000 == 0:
                            db.commit()
                            print(f"SFT cache: encoded={count} selected={statistics['selected']}", flush=True)
                    db.execute("insert into groups values (?,?,?)", (start, count, max(1., rate)))
                    db.commit()
                if not count:
                    raise ValueError("No SFT records selected")
                db.execute("drop table raw")
                statistics["cached"] = count
                db.execute("insert into metadata values ('statistics',?)", (json.dumps(statistics),))
                db.execute("insert into metadata values ('complete',?)", (self.key,))
            temporary.replace(self.path)
            print("SFT cache complete: "+json.dumps(statistics), flush=True)
        finally:
            temporary.unlink(missing_ok=True)

    def start_epoch(self, epoch, rank=0, world=1):
        self.build()
        with sqlite3.connect(self.path) as db:
            if not db.execute("select count from epochs where epoch=?", (epoch,)).fetchone():
                groups = list(db.execute("select start,stop,rate from groups order by rowid"))
                lengths = dict(db.execute("select id,length from records"))
                items = ((index, lengths[index]) for index in epoch_indices(groups, epoch, self.seed))
                count = 0
                for pack in greedy_pack(items, self.processor.max_length, self.disable_packing):
                    db.execute("insert into plans values (?,?,?)", (epoch, count, json.dumps(pack)))
                    count += 1
                db.execute("insert into epochs values (?,?)", (epoch, count))
        self.epoch, self.cursor, self.rank, self.world = epoch, rank, rank, world

    def next_records(self):
        with sqlite3.connect(self.path) as db:
            result = db.execute("select indices from plans where epoch=? and id=?", (self.epoch, self.cursor)).fetchone()
            if result is None:
                return None
            records = [json.loads(db.execute("select payload from records where id=?", (index,)).fetchone()[0])
                       for index in json.loads(result[0])]
        self.cursor += self.world
        return records

    def state_dict(self):
        return {"key": self.key, "epoch": self.epoch, "cursor": self.cursor, "rank": self.rank, "world": self.world}

    def load_state_dict(self, state):
        if state["key"] != self.key or state["rank"] != self.rank or state["world"] != self.world:
            raise ValueError("Data/config/rank/world-size changed; cannot resume the same stream")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
