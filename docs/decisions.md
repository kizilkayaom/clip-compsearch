# Decisions (Stage 1)

Each entry records a choice made before any test-set evaluation.

## Step 2: Split (13 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Validation slice | 5,000 train2017 images | Same gallery size as the val2017 test, so validation Recall@K is on the same scale. |
| Split seed | 0 | Any fixed value works. It is recorded in `results/split_report.json`. |
| Caption rule for training | Keep all captions. In each epoch, sample 1 caption per image with the run seed. | No image appears twice in a batch, so no caption is a false negative in the contrastive loss. All captions are used across epochs. |
| Test set | All 5,000 val2017 images | SugarCrepe uses val2017 images. |

## Step 4: Negatives (13 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Caption cleaning | Collapse whitespace runs to 1 space and strip the ends | 84,106 train2017 captions have leading or trailing whitespace, and 6,889 have internal double spaces. A double space before "of" broke the "of" rule. CLIP's tokenizer gives the same token IDs either way (tested). |
| Compound nouns | Merge each compound modifier with its head noun into one unit, and swap whole units. | On 2,000 captions, 25% of swaps split a compound ("living room" → "room living"). |
| Filter | Rule filter: reject a swap when the 2 units are joined by "and"/"or" or by next to, beside, near, by, alongside, across from, opposite. | NLI (DeBERTa-v3-small) labelled 52% of swaps as entailment, including real negatives, and it kept the dog/cat coordination swap. |
| NLI scores | Stored per pair, not used for filtering | Keeps the evidence for later analysis. |
| Articles | Fix "a"/"an" directly before each swapped unit | "A woman with an umbrella" → "An umbrella with a woman". |
| Run location | Local Mac (MPS) | Measured about 437 captions/s for spaCy and 286 pairs/s for NLI. |

## Step 4 review (13 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| 3 rare parse cases | Record as known limits | 3 of 1,802 kept pairs (0.17%), below the 3% stopping rule. |
| `data/negatives/` | Ignored by git | The report stores the SHA-256 of each file, and the script rebuilds the files. |
| `docs/decisions.md` | Tracked (`docs/*` ignored, `!docs/decisions.md` re-included) | It records why each method choice was made. |
| Bootstrap unit (step 11) | Resample by image ID, for SugarCrepe and for Recall@K | SugarCrepe has 7,511 examples on 1,560 images, and each COCO image has about 5 captions, so examples of one image are not independent. |
| Labelling | Add a `break_type` column: "article or number", "word order", "other" | Shows whether the unfixed plural agreement passes the 3% limit. |
| Reproducibility hash | Report `pairs_sha256` over image_id, caption_id, caption, negative and swap_type, next to the file `sha256` | Rounded NLI scores differ between MPS and CPU (8 of 150 pairs) and between batch sizes (4 of 150), so the file hash does not reproduce on another device. |

## Open decisions

- Selection rule (step 6)
- Logit scale: fixed or trainable (step 7)
- Batch size and learning-rate grid (step 7)
- Approval of the time and cost estimate (step 8)
- Where to cache the image embeddings and where to train (after the step 8 timing)
