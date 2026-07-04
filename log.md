# CLIP-CompSearch — Build Log

## Day 1 — Jul 2, 2026
- Environment: venv created, torch + transformers + pillow installed
- Verified MPS backend
- Completed 2 tensor drills
  - L2-normalization, matmul and similarity matrices, top-k ranking
- Built a working text-to-image search loop
  - 5 queries with correct top-1 ranking

### What I learned?
`get_image_features` returns `BaseModelOutputWithPooling`, so we need to extract the `.pooler_output` for the tensor 

## Day 2 - Jul 3, 2026
- Cached image embeddings to disk by using `torch.save`, verified identical search results on reload
- Refactored search logic `engine.py` in (divided into 6 functions: `load_model`, `load_cached_embeddings`, `load_text_data`, `normalize`, `construct_sim_matrix`, `extract_topk`)
- Wrote the calling script to verify the module thoroughly and verified that the results match across all three runs

## Day 3 - Jul 4, 2026
- Built compositional accuracy harness working on SugarCrepe's `swap_obj` data
- Baseline result: 61.2% accuracy on `swap_obj` (245 examples), which confirms the high rate of failure on bag-of-words.
- Reversed verification: Flipped accuracy comparison produces the complementing rate (38.8%) which proves that the harness method is working correctly.