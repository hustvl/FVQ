import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unittest

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training"))
from processing import TrainingProcessor, Overlength
from cache import SFTCache, epoch_indices, greedy_pack, tokenizer_fingerprint
from stream import PretrainStream
from fvq.processing import ImageProcessor


class ToyTokenizer:
    eos_token_id, pad_token_id = 2, 2
    def __init__(self):
        tokens = list(TrainingProcessor.ROLE_TOKENS.values())+list(TrainingProcessor.SEPARATORS)
        tokens += ["<img_start>", "<img_end>", "<img_pad>"]
        self.vocab = {token: index+10 for index, token in enumerate(tokens)}
        self.pattern = re.compile("|".join(map(re.escape, tokens)))
        self.backend_tokenizer = self
    def get_vocab(self):
        return self.vocab
    def to_str(self):
        return json.dumps({"model":"toy-tokenizer-v1", "truncation":getattr(self,"truncation",None)})
    def convert_ids_to_tokens(self, index):
        return next(token for token, value in self.vocab.items() if value == index)
    def encode(self, text, add_special_tokens=False):
        result, end = [], 0
        for match in self.pattern.finditer(text):
            result.extend(ord(char)+100 for char in text[end:match.start()])
            result.append(self.vocab[match.group()]); end = match.end()
        result.extend(ord(char)+100 for char in text[end:])
        return result
    def decode(self, ids, skip_special_tokens=False):
        reverse = {value: key for key, value in self.vocab.items()}
        return "".join(reverse[token] if token in reverse else chr(token-100) for token in ids)


class DataTests(unittest.TestCase):
    def processor(self, length=128):
        return TrainingProcessor(ToyTokenizer(), ImageProcessor(min_pixels=1024, max_pixels=1024), length)

    def test_direct_sft_json_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sft.json'
            rows = [{'messages':[{'role':'user','content':f'Q{i}'},
                                 {'role':'assistant','content':f'A{i}'}]} for i in range(4)]
            path.write_text(json.dumps(rows))
            cache = SFTCache(path, self.processor(), Path(tmp)/'cache', seed=42)
            cache.start_epoch(0)
            records = cache.next_records()
            self.assertEqual(len(records), 4)
            self.assertTrue(all(any(label >= 0 for label in row['labels']) for row in records))
            path.write_text(json.dumps({'unexpected':'not a manifest'}))
            with self.assertRaises(KeyError):
                SFTCache(path, self.processor(), Path(tmp)/'cache')

    def test_demo_data_formats_and_no_overwrite(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
        from make_demo_data import make_demo
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'demo'
            make_demo(root, records=8)
            proc = self.processor(length=1024)
            for name, mode in [('pt.jsonl','pretrain'), ('text_pt.jsonl','pretrain'),
                               ('pt_conversations.jsonl','pretrain'), ('sft.json','sft'), ('text_sft.json','sft')]:
                path = root/name
                rows = ([json.loads(line) for line in path.read_text().splitlines()]
                        if path.suffix == '.jsonl' else json.loads(path.read_text()))
                self.assertEqual(len(rows), 8)
                for row in rows:
                    record = getattr(proc, mode)(proc.normalize(row, root))
                    self.assertTrue(any(label >= 0 for label in record['labels']), name)
                    if 'images' in row:
                        self.assertEqual(len(record['grids']), len(row['images']))
            self.assertEqual(len((root/'tokenizer.jsonl').read_text().splitlines()), 8)
            with self.assertRaises(FileExistsError):
                make_demo(root, records=8)
            with self.assertRaises(ValueError):
                make_demo(Path(tmp)/'bad', records=0)

    def test_tokenizer_cache_key_ignores_temporary_truncation(self):
        tokenizer = ToyTokenizer()
        before = tokenizer_fingerprint(tokenizer)
        tokenizer.truncation = {"max_length":128}
        self.assertEqual(before,tokenizer_fingerprint(tokenizer))

    def test_pretrain_spans_eosloss_and_continuous_packing(self):
        proc = self.processor()
        row = proc.normalize({"messages": [{"role":"user", "content":"Q"},
                  {"role":"assistant", "content":"Answer"}], "eosloss":False}, ".")
        record = proc.pretrain(row)
        self.assertEqual(record["labels"], [-100]+record["ids"][1:-1]+[-100])
        plain = proc.pretrain({"title":"T", "input":"body"})
        self.assertEqual(plain["ids"], plain["labels"])
        batch = proc.batch([record, plain], "pretrain")
        self.assertNotIn("position_ids", batch)
        self.assertNotIn("sequence_lengths", batch)
        self.assertEqual(batch["input_ids"][0,-1], proc.tokenizer.pad_token_id)

    def test_sft_roles_cot_and_explicit_learnable(self):
        proc = self.processor()
        row = {"conversations":[{"from":"human", "value":"Q", "cot":True},
              {"from":"gpt", "value":"<think>R</think>A", "learnable":False},
              {"from":"human", "value":"B", "learnable":True}]}
        encoded = proc.sft(proc.normalize(row, "."))
        self.assertEqual(encoded["ids"][0], proc.roles["user"])
        self.assertEqual(encoded["ids"][-1], proc.tokenizer.eos_token_id)
        self.assertEqual([x for x in encoded["labels"] if x >= 0], proc.tokenizer.encode("B")+[2])
        altered = json.loads(json.dumps(row)); altered["conversations"][0]["cot"] = False
        self.assertEqual(encoded, proc.sft(proc.normalize(altered, ".")))
        ordinary = proc.sft({"messages":[{"role":"user", "content":"Q"}, {"role":"assistant", "content":"A"}]})
        packed = proc.batch([ordinary, ordinary], "sft")
        n = len(ordinary["ids"])
        self.assertEqual(packed["position_ids"][0,:n*2].tolist(), list(range(n))*2)
        self.assertEqual(packed["sequence_lengths"], (n,n,128-2*n))
        self.assertEqual(packed["input_ids"][0,-1], 0)

    def test_media_and_fail_closed(self):
        proc = self.processor()
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (32,32)).save(Path(tmp)/"a.png")
            row = {"images":["a.png"], "messages":[{"role":"user", "content":"<image>Q"},
                     {"role":"assistant", "content":"A"}]}
            record = proc.sft(proc.normalize(row, tmp))
            batch = proc.batch([record], "sft")
            self.assertEqual(record["grids"], [[1,2,2]])
            self.assertEqual(batch["pixel_values"].shape, (4,1536))
            self.assertEqual((batch["input_ids"]==proc.image_ids[2]).sum(), 1)
            typed = {"messages":[{"role":"user", "content":[{"type":"image", "text":"a.png"},
                                 {"type":"text", "text":"Q"}]}, {"role":"assistant", "content":"A"}]}
            self.assertEqual(record, proc.sft(proc.normalize(typed, tmp)))
        with self.assertRaisesRegex(ValueError, "Only local"):
            proc.normalize({"content":"<img_start>https://example.org/a.png<img_end>"}, ".")
        with self.assertRaisesRegex(ValueError, "no supervised"):
            proc.sft({"messages":[{"role":"user", "content":"Q"}]})
        with self.assertRaises(Overlength):
            self.processor(4).sft({"messages":[{"role":"user", "content":"long"}]})

    def test_epoch_sampling_and_pack(self):
        groups = [(0,10,1.5), (10,20,2.)]
        a = epoch_indices(groups, 0, 42); b = epoch_indices(groups, 1, 42)
        self.assertEqual(len(a),35)
        self.assertEqual(sorted(x for x in a if x<10), sorted(list(range(10))+list(range(5))))
        self.assertEqual(sorted(x for x in b if x<10), sorted(list(range(10))+list(range(5,10))))
        self.assertEqual(list(greedy_pack([(0,3),(1,4),(2,4)],7)), [[0,1],[2]])

    def test_sft_cache_reuse_rates_resume_and_errors(self):
        proc = self.processor(32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                path = root/f"{name}.jsonl"
                path.write_text("".join(json.dumps({"messages":[{"role":"user","content":name+str(i)},
                                     {"role":"assistant","content":"ok"}]})+"\n" for i in range(10)))
            manifest = root/"mix.json5"
            manifest.write_text('{train_files:{root:".", sample_rate:{"a.jsonl":0.5,"b.jsonl":1.5}}}')
            cache = SFTCache(manifest, proc, root/"cache")
            cache.start_epoch(0)
            timestamp = cache.path.stat().st_mtime_ns
            first = cache.next_records()
            state = json.loads(json.dumps(cache.state_dict()))
            rest = []
            while (records := cache.next_records()) is not None:
                rest.append(records)
            clone = SFTCache(manifest, proc, root/"cache")
            clone.start_epoch(0); clone.load_state_dict(state)
            self.assertEqual([clone.next_records() for _ in rest], rest)
            self.assertIsNone(clone.next_records())
            self.assertEqual(cache.path.stat().st_mtime_ns, timestamp)
            self.assertTrue(first)
            with sqlite3.connect(cache.path) as db:
                self.assertEqual(db.execute("select count(*) from records").fetchone()[0],15)
                self.assertEqual(list(db.execute("select start,stop,rate from groups")),[(0,5,1.),(5,15,1.5)])
            (root/"a.jsonl").write_text('{bad}\n')
            broken = SFTCache(manifest, proc, root/"cache")
            self.assertNotEqual(broken.key, cache.key)
            with self.assertRaisesRegex(ValueError, "line 1"):
                broken.build()
            self.assertFalse(broken.path.exists())

    def test_stream_chunks_sharding_and_resume(self):
        proc = self.processor(80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/"train.jsonl"
            path.write_text("".join(json.dumps({"content":f"row-{i}"})+"\n" for i in range(90)))
            streams = [PretrainStream(path,proc,root/"index",r,2,chunk_bytes=64,pool_size=9) for r in range(2)]
            seen = []
            for data in streams:
                while (pack := data.next_records()) is not None:
                    seen.extend(proc.tokenizer.decode(record["ids"][:-1]) for record in pack)
            self.assertEqual(sorted(seen),sorted(f"row-{i}" for i in range(90)))
            data = PretrainStream(path,proc,root/"index2",chunk_bytes=64,pool_size=9)
            data.next_records()
            state = json.loads(json.dumps(data.state_dict()))
            rest = []
            while (pack := data.next_records()) is not None:
                rest.append(pack)
            clone = PretrainStream(path,proc,root/"index2",chunk_bytes=64,pool_size=9)
            clone.load_state_dict(state)
            self.assertEqual([clone.next_records() for _ in rest], rest)
            self.assertIsNone(clone.next_records())

    def test_counted_overlength_filter_and_incompatible_resume(self):
        proc = self.processor(16)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/"sft.jsonl"
            row = {"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]}
            long = {"messages":[{"role":"user","content":"x"*40},{"role":"assistant","content":"A"}]}
            path.write_text("\n".join(map(json.dumps,[row,row,long]))+"\n")
            cache = SFTCache(path,proc,root/"cache",overlength="filter")
            cache.start_epoch(0)
            with sqlite3.connect(cache.path) as db:
                stats = json.loads(db.execute("select value from metadata where key='statistics'").fetchone()[0])
            self.assertEqual((stats["cached"],stats["duplicates"],stats["overlength"]),(1,1,1))
            state = cache.state_dict(); state["world"] = 2
            with self.assertRaisesRegex(ValueError,"world-size changed"):
                cache.load_state_dict(state)
            path = root/"pt.jsonl"
            path.write_text(json.dumps({"content":"x"*40})+"\n"+json.dumps({"content":"ok"})+"\n")
            stream = PretrainStream(path,proc,root/"index",overlength="filter")
            self.assertEqual(len(stream.next_records()),1)
            self.assertEqual(stream.filtered,1)


if __name__ == "__main__":
    unittest.main()
