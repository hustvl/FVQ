# Validation scope

The following tests use independently loaded legacy and standalone models,
not two calls into the same implementation. Trained weights, source locations,
test images and raw internal reports are kept outside this code directory.

| Check | Tested scope | Result |
| --- | --- | --- |
| Preprocessing | Five aspect ratios/resolutions, including 1024×1536 | Packed FP32 pixels and grids exactly equal |
| Trained tokenizer | EMA checkpoint at step 600,000; 224×224, 384×256, 512×512 and packed two-image inputs | Zero group-ID mismatches; encoder, quantized and decoder features and quantization loss max difference 0 |
| FVQ updates | Three independent CUDA BF16 optimizer steps, including forced dead-code restart | IDs, outputs, losses, gradients, state tensors and global EMA exactly equal |
| Trained 8B VLM | Text, one image and two images | Zero ID mismatches; visual embeddings, all 37 hidden-state outputs and logits max difference 0; greedy output tokens identical |
| CPU unit tests | FVQ state/precision/ties, packed attention, data formats, sampling, packing, cache, malformed inputs, resume, scheduler, config paths and launcher argv | Nineteen tests passed |
| Stage launchers | Tokenizer, Alignment, PT and SFT; both caption and conversation PT mixtures tested with a small model and saved snapshots | Finite losses; checkpoints/training states saved; FVQ frozen throughout VLM stages; Alignment LLM unchanged; later LLM and bridge weights updated |
| Four-rank example data | Caption/conversation PT and SFT mixture formats, generated synthetic data and actual exported text-tokenizer vocabulary | Every rank received a supervised first pack; SFT JSON-array cache built successfully |
| Independent PT preprocessing | Six text/message/image records and a packed batch | Tokens, labels, pixels and grids exactly equal |
| Independent SFT preprocessing | Five role/think/learnable/multi-image records | Tokens, labels, pixels and grids exactly equal |
| Independent data selection | SFT rates 0.5 and 1.5, three epochs with four packs each; two-rank PT byte-chunk indices | Selection, epoch plans, packs and chunk indices exactly equal for the tested clean inputs |
| Trained 8B packed training | PT and SFT two-image forward/backward with frozen language/vision weights | Finite losses 2.783678 and 4.781105; finite bridge gradients; tokenizer IDs unchanged |
| Four-GPU data resume | Small model with the trained text vocabulary, BF16, ZeRO-2, gradient accumulation 2; PT and SFT resumed from step one | Step-two model tensors, all four optimizer shards and all four data states exactly equal to uninterrupted training |
| Public vision initialization | Local public Qwen3-VL-8B vision weights | Strict load and tokenizer export passed |
| Tokenizer training CLI | Real trained weights, one optimizer step and state resume to step two | Finite loss; live weights, global EMA and training state saved |
| Full 8B training CLI | Four H100 GPUs, ZeRO-2, one SFT optimizer step | Finite loss 6.108902; model weights saved |
| Installed-wheel inference | New virtual environment with existing CUDA dependencies, no original repository imports | Trained tokenizer EMA and one-step VLM weights reloaded; finite outputs and image generation passed |
| Installed-wheel data path | Portable SFT cache, saved small-model checkpoint and CPU packed forward | Finite loss; no original repository imports |

CPU unit-test statement coverage is 86% (394/458 statements) for the `fvq`
package and 89% (460/515) for the cache, training processor and PT stream.
Across the package and all training modules it is 74% (919/1241); CLI trainers
are exercised separately by integration tests, not this unit coverage run.
These are statement counts, not GPU branch coverage. No claim of complete
test coverage is made.

Source files and extracted wheel/source-distribution contents were checked
with the local release heuristic and Detect Secrets (without network
verification); no unresolved findings remained after review. Relative example
data paths were reviewed as heuristic false positives. The three paper PNG assets
were visually reviewed and checked for text metadata/attachments. Bandit had no
unresolved findings. Narrow suppressions cover local-only model asset loading,
non-cryptographic data shuffling and a test vocabulary token ID. These checks
supplement, not replace, the company's approved security/static review.

The reference environment was Python 3.12, PyTorch 2.6/CUDA 12.4,
Transformers 4.57.0, BF16 vision computation and H100 GPUs. Exact equality
is established only for the tested inputs/environment, not every kernel,
hardware, resolution or dependency version. Language inference uses math SDPA
and legacy FP32 rotary arithmetic. Packed SFT GPU training uses variable-length
FlashAttention, with isolated per-conversation attention;
full VLM training trajectories are not claimed bit-identical to the legacy repo.

These are functional and numerical smoke tests, not a new benchmark evaluation
or proof of long-run convergence. The release targets the verified Qwen3-VL
vision/G16 FVQ/concatenated-embedding/1D-RoPE checkpoint family only.
Interleaved-MRoPE, alternate vision backbones, pixel reconstruction and
audio/video are outside the tested release boundary.

Tokenizer optimizer/EMA resume still restarts its bounded image JSONL stream.
VLM stages start fresh optimizer/scheduler state from model weights unless
`--resume` is used. VLM resume restores optimizer/scheduler/RNG and next-consumed
per-rank data state and requires unchanged configuration, data and world size.
Distributed epoch tails are dropped jointly to avoid incomplete accumulation or
rank hangs. Portable caches can be reused across world sizes, but an in-progress
data cursor cannot. Private storage clients, business-specific data filters,
legacy RecordIO caches and experiment-specific exclusion indices are not copied.
PT serial ordering is deterministic rather than preserving thread-completion
order. SFT retains explicit `learnable` labels that the legacy cache discarded.
See DATA.md for these intentional boundaries; full training-trajectory equality
is not claimed.

No project license or company release approval is implied by these tests.
Required dependency notices are retained. Project licensing, ownership review,
approved static/security analysis and publication approvals remain separate.
