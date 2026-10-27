# Figure notes

- `reconstruction_vs_understanding.png`: Figure 2 (left), cropped from the original
  manuscript figure. Compares VQ with RVQ-Odd and illustrates retaining odd-group
  indices for VLM training. Higher feature-reconstruction error does not imply
  worse multimodal understanding.
- `fvq_method.png`: Figure 3(a-b). Overview of FVQ, including the training pipeline
  and the factorized quantizer. During quantizer training, multiple dense
  projections produce complementary views of the same visual feature, which are
  quantized in parallel by independent codebooks. In the subsequent VLM training
  stage, the quantizer is frozen, and only the resulting discrete indices are
  mapped to newly initialized token embeddings for multimodal understanding.
- `bit_budget_scaling.png`: Figure 1(a-b). Average multimodal understanding
  performance improves with the per-patch bit budget. FVQ continues to improve in
  the high-budget regime and reaches the continuous QwenViT baseline, whereas
  existing multi-codebook quantizers saturate. Higher budgets progressively recover
  the fine-grained textual information required for correct predictions; green and
  red denote correct and incorrect answers, respectively.

These assets are rasterized from the manuscript for Markdown
display. PDF/PPTX
sources, hidden document metadata, standalone benchmark files, private paths and raw
evaluation records are not distributed here. The visible plots and diagrams
were reviewed. Inclusion in these figures does not grant redistribution rights
for any underlying dataset.
