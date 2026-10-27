# Local data templates

These are synthetic format examples, not benchmark or private training records.
The referenced image is intentionally not bundled. To create a local red-square
image and usable demo sources, run `python tools/make_demo_data.py` from the
repository root. It writes `data/demo` and refuses to overwrite existing data.

| Stage | Format | Template |
| --- | --- | --- |
| Tokenizer | One JSON object per line | `tokenizer.jsonl` |
| Alignment / PT | One JSON object per line, inline image paths and trainable spans | `pt.jsonl`, `text_pt.jsonl` |
| PT (conversation mixture) | One JSON object per line, conversations normalized to pretrain spans | `pt_conversations.jsonl` |
| SFT | JSON array or JSONL, assistant targets | `sft.json`, `text_sft.json` |

Copy the relevant format, use your own images/text, and set `data` in the stage
YAML to your file. PT needs JSONL for byte-chunk streaming; a JSON array is not
accepted for these stages. SFT accepts a JSON array directly without a manifest.
For multiple sources, edit the corresponding `configs/*_data.json5` manifest.

Image paths are relative to their source file; the file must exist. Conversation records need
one `<image>` marker per image in encounter order. Plain text needs no image.
Only include `<think>...</think>` when that is actually in your targets; `cot`
metadata alone does not generate a reasoning answer. Malformed samples fail.

The generated defaults repeat toy content with distinct example prefixes so
there are enough records for a four-rank format smoke test. They are not intended
for learning useful model capabilities. Tiny custom datasets can exhaust before
every rank receives a pack; use one process for tiny data, or enough real records
for your distributed process count and packing length.
