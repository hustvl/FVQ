# Data pipeline contract

## Formats and media

Tokenizer learning uses a JSONL containing `{"images":["images/example.png"]}`
per line. PT requires JSONL for byte-chunk streaming. SFT accepts JSONL or a
JSON/JSON5 array of conversations. A single SFT data file can be passed directly;
multiple sources use the `train_files` mixture manifest below.
See `examples/data/` for complete synthetic records and `configs/*_data.json5`
for caption/conversation PT mixtures and SFT mixtures. All four stage YAMLs expose the runtime
`checkpoint`, `data`, `output` paths; VLM YAMLs also expose `cache`. CLI flags
override these values. No model or loader code editing is necessary.

All paths are runtime inputs, not embedded dataset inventories. Source paths in
a mixture manifest are relative to its `train_files.root`, itself relative to
the manifest; image paths are relative to the source data file. An absolute path is
also accepted. A source may be an exact file, glob or historical prefix match.

PT accepts `{"title":"...","input":"..."}`, `{"content":"..."}`,
list records with text at index 1, or nonempty messages. Messages accept
`role/content` or `from/value`, with human/gpt aliases. User/system content is
conditioning; assistant content becomes a trainable span. Unmarked plain text
is fully trainable. Explicit `<trainable_start>...<trainable_end>` spans and
`eosloss: false` are supported, including the original whole-span rule.

SFT accepts messages/conversations, role aliases, typed text/image lists and
explicit boolean `learnable`. It uses the saved vocabulary's `<C_Q>`, `<C_A>`,
`<B_SYS>` and associated role/separator tokens, not a new chat template.
`user_system` uses the system role token. Supervised turns train their content
and terminal EOS; user/system content is otherwise masked. Role/image special
token masking follows the model export. The tokenizer must contain the verified
training vocabulary; arbitrary base-model vocabularies are not silently remapped.

Images can be inline local paths, local descriptors, or `images` with one
`<image>` marker per image. The cache stores token IDs/labels, image references
and grids, not decoded pixels. Pixel loading is lazy and checked against cached
grid sizes. Remote object-store descriptors are rejected explicitly: mount or
materialize them, or supply a separately audited storage integration. Proprietary
storage endpoints/credentials and audio/video paths are not included.

## Sampling and packing

The JSON5 manifest uses `train_files: {root, sample_rate, stream_sample}`.
Rates are source multipliers, **not automatically token percentages**.

- PT samples line-aligned 32 KiB byte chunks. Initial file ordering, chunk
  assignment by rank and seeded chunk sampling match the original index logic
  when an explicit sampling seed is used. Each read pool is shuffled and greedily
  packed; EOS is appended per record. Its original training entry discards
  reset positions and sequence boundaries, so the public trainer also uses
  continuous positions and causal attention. Deterministic serial processing
  replaces nondeterministic thread-completion order.
- SFT rates below 1 select `floor(N*rate)` after seeded shuffle, once before
  tokenization. Optional `stream_sample: true` uses seeded Bernoulli selection.
  Rates at least 1 retain each record once in the cache; each epoch repeats the
  integer part and rotates the fractional window, shuffles with `seed+epoch+1`,
  then greedily packs in that order. Conversations have reset positions and
  isolated attention. Zero padding is retained for SFT, tokenizer pad for PT.

The SFT SQLite cache is keyed by source size/mtime, tokenizer, resolution,
max length, sampling settings and implementation. Tokenization is built once;
epoch plans are cached separately. The same completed cache can serve different
GPU counts. Cache **resume cursors** require unchanged world size. Exact record
order is tested for clean inputs, not asserted for every private filtering rule.
Global deduplication uses canonical normalized conversation content, not unsafe
pickle deserialization. Explicit `learnable` is retained in cached labels; the
original cache's loss of this flag is intentionally not replicated.

Malformed JSON, missing content/media, unsupported roles and no-target SFT fail
with a source/location reference, not a raw record dump. There is no random
replacement/infinite retry or hidden business-specific content filter.
Overlength filtering is opt-in and reports counts. This changes how dirty inputs
are handled; sanitize data explicitly rather than expecting the original private
policy filters. The original serial PT EOS-label bug is not replicated.

## State and privacy

Training snapshots save per-rank next-consumed state, including PT pending packs
and shuffle RNG, plus Accelerate model/optimizer/scheduler/RNG state. No background
reader can advance the saved pointer beyond consumed training. A new stage only
loads model weights; `--resume` requires the same configuration, data and world size.

Keep generated `cache/`, `outputs/`, data-state JSON, optimizer dumps and data
files out of a release. They can contain local paths and private sample text.
Only source, relative example configs and synthetic tests are included here.
Original RecordIO caches and experiment-specific exclusion indices are not
automatically imported; rebuild the portable cache from the selected local data.
