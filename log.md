# CLIP-CompSearch — Build Log

## Day 1 — Jul 2, 2026
- Environment: venv created, torch + transformers + pillow installed
- Verified MPS backend
- Completed 2 tensor drills
  - L2-normalization, matmul and similarity matrices, top-k ranking
- Built a working text-to-image search loop
  - 5 queries with correct top-1 ranking
-

### What I learned?
`get_image_features` returns `BaseModelOutputWithPooling`, so we need to extract the `.pooler_output` for the tensor 