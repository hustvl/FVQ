# Third-party components

The vision encoder and language-model backbone use Hugging Face Transformers
4.57.0: https://github.com/huggingface/transformers/tree/v4.57.0
Transformers is licensed under Apache License 2.0; its license is reproduced in
`licenses/transformers-LICENSE`. The attention adaptation in `src/fvq/vlm.py`
follows the Qwen3 attention tensor layout, adding FP32 rotary arithmetic and
legacy math-SDPA inference. Other backbone implementation remains in the
installed Transformers dependency.

The pretrained vision encoder comes from Qwen/Qwen3-VL-8B-Instruct, whose
official model card specifies Apache License 2.0:
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
The language-model backbone and text tokenizer come from Qwen/Qwen3-8B,
also under Apache License 2.0:
https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE
Model weights and tokenizer assets are not bundled with this source tree.
Weight releases must retain applicable upstream licenses, attribution notices
and modification notices; the FVQ license does not remove those obligations.

PyTorch, FlashAttention,
Accelerate, DeepSpeed, Safetensors, Pillow, NumPy, PyYAML and JSON5 retain their own licenses.
The FVQ implementation uses Apache License 2.0; see the root LICENSE file.
