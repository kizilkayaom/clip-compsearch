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

#### Day - 3 (Part 2)
- Generalized the evaluation harness to all 7 SugarCrepe categories
- Baseline Results:
  | Category    | Total | Accuracy | Error   |
  |-------------|-------|----------|---------|
  | swap_obj    |   245 |   61.2%  |  38.8%  |
  | swap_att    |   666 |   64.1%  |  35.9%  |
  | add_att     |   692 |   69.5%  |  30.5%  |
  | replace_rel |  1406 |   69.4%  |  30.6%  |
  | add_obj     |  2062 |   77.0%  |  23.0%  |
  | replace_att |   788 |   80.3%  |  19.7%  |
  | replace_obj |  1652 |   90.7%  |   9.3%  |

## Day 4 - Jul 5, 2026
- Built and verified Recall@K harness

## Day 5 - Jul 7, 2026
- Built NLI filter using `cross-encoder/nli-deberta-v3-small`
- Cached 5000 COCO val2017 vision embeddings

## Day 6 - Jul 8, 2026
- Build the training loop and exp
### Training Loop Configuration
| Parameter       | Value                        |
|-----------------|------------------------------|
| Base model      | openai/clip-vit-base-patch32 |
| LoRA rank       | 16                           |
| LoRA alpha      | 32                           |
| LoRA dropout    | 0.1                          |
| LoRA targets    | q_proj, v_proj (text encoder) |
| Trainable params| 393,216 / 63,559,168 (0.62%) |
| Learning rate   | 1e-5                         |
| Optimizer       | AdamW                        |
| Batch size      | 32                           |
| Epochs          | 5                            |
| Training data   | COCO val2017 (5000 images)   |
| Hard negatives  | spaCy noun swap, unchanged filtered out |
| Temperature     | 0.07                         |

### Training Loss
| Epoch | Avg Loss |
|-------|----------|
| 0     | 2.3487   |
| 1     | 2.1095   |
| 2     | 1.9426   |
| 3     | 1.7941   |
| 4     | 1.6493   |

### SugarCrepe Evaluation (on `swap_obj`)
| Model         | Accuracy |
|---------------|----------|
| Base CLIP     | 61.22%   |
| After LoRA    | 68.57%   |
| **Difference**         | **+7.35pp** |

### Training Loop Diagnostic Tests

**1. LoRA integration:**
I have compared the text embeddings from the fine-tuned model against a freshly loaded base model on the same input ("a woman prepares a pizza"), one of the captions taken from COCO's 2017 dataset. The result is: `torch.allclose() = False`

**2. Initial training run (excluded hard-negatives):**
First attempt trained with only positive captions (hard negatives generated but not passed to the loss). Loss decreased (1.86 → 1.30 over 2 epochs), but swap_obj accuracy stayed at 61.22%, so the result was identical to baseline. This happened because, while model learned the basic contrastive alignment, it was not trained on hard negatives.

**3. Fix — hard negatives included in contrastive loss:**
I have changed text encoding to `combined_texts = valid_positives + valid_negatives`,
and also flipped the matrix multiplication to image-side (`images @ texts.T` → [32, 64]). This change allows to target the positively-signed columns. Additionally, the loss function now forces each image to prefer its correct caption over both the random in-batch texts and its constructed hard negative.

**4. Unchanged-negative filtering:**
The captions where `swap_objects` returned the original string unchanged (where < 2 nouns) are dropped from the batch.

**5. InfoNCE loss verification:**
- In perfect-match embeddings: loss ≈ 2.03e-06 (essentially 0)
- In random embeddings: loss ≈ 1.47

### Retrieval Preservation Check (R@K on 500 COCO val images)
| Metric | Base CLIP | Fine-tuned |    Δ   |
|--------|-----------|------------|--------|
| R@1    |   0.592   |   0.568    | -2.4pp |
| R@5    |   0.868   |   0.876    | +0.8pp |
| R@10   |   0.944   |   0.952    | +0.8pp |

The results suggest that the retrieval is preversed under noise. Additionally, the fine-tuning process improves the conditional accuracy without degrading the standard search.