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

## Step 5: Image embeddings (14 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Location | Local Mac (Apple M4, MPS) | Measured about 143 images/s, so about 15 minutes for 123,287 images. No 18 GB upload. |
| Feature | `get_image_features(...).pooler_output`, not normalized, float32 | In transformers 5.12.1 this equals `visual_projection(vision_model(...).pooler_output)` (tested). One function in `src/features.py` for caching, training and evaluation. |
| Preprocessing | PIL `convert("RGB")`, then `CLIPProcessor` | 10 val2017 images are not RGB. |
| File format | 1 file per set: `image_ids` and `features` in the same `.pt` file | The rows and IDs cannot drift apart. |
| Sanity check | 100 seeded images per set, recomputed on CPU with batch size 1; tolerance 1e-4 max abs difference | Measured MPS vs CPU difference: up to 3.4e-05 on norms of 9.2–11.6. A swapped row gives a difference above 1e-2 (tested). |
| Parallelism | 8 threads in the main process, with prefetch of the next batch | DataLoader worker processes ran at 33–57 images/s on macOS, against 132–143 images/s with threads. |

## Step 6: Validation metrics and selection rule (14 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Compositional accuracy | 1 caption per validation image with a kept pair (4,986 images), chosen with seed 0. Correct when cos(image, caption) > cos(image, negative); a tie counts as wrong. | Copies the SugarCrepe format without SugarCrepe data. |
| Retrieval | Text-to-image Recall@1/5/10 over the 5,000 validation images and all 25,012 captions, matched by image ID. Ties count against the true image. | Same gallery size as the val2017 test. |
| Text input | Truncate to 77 tokens | CLIP's text encoder has 77 positions. |
| Selection rule | Consider only epochs (and learning rates) whose validation R@1 is at most 1 point below base CLIP. From those, choose the highest compositional accuracy. If none passes, choose the highest R@1 and report it. | Keeps retrieval while improving composition. Chosen before any training run. |
| Known bias | The validation negatives come from the same generator as the training negatives, so compositional accuracy favours the hard-negative arm. | SugarCrepe in step 11 has no such bias. |

## Step 7: Training setup (14 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Arms | B `inbatch`: in-batch negatives only. C `hard`: plus 1 hard negative per caption as extra image-to-text columns. | Same images, captions, batches, steps and hyperparameters; only the extra negatives differ. |
| Training pairs | Only images with a kept negative (113,019 of 113,287), 1 caption per image per epoch, chosen and shuffled with `Random(f"{seed}:{epoch}")`; last partial batch dropped | Both arms see identical batches (tested with a batch hash). |
| Loss | Symmetric InfoNCE: mean of image-to-text and text-to-image cross-entropy. Hard negatives only in the image-to-text direction. | Plan step 7. |
| Logit scale | Fixed at CLIP's learned value (exp(4.6052) = 100.0), not trained | The features were trained at this scale; one less variable between arms. |
| LoRA | r=16, alpha=32, dropout 0.1 on `q_proj` and `v_proj` of the text encoder only (393,216 parameters) | Same as the original project. |
| Optimizer | AdamW, weight decay 0.01, batch size 256, linear warmup over 50 steps, then constant | A constant rate keeps a resumed run identical to a straight run (tested), which the epoch-extension rule needs. |
| Learning-rate grid | 2e-5, 1e-4, 5e-4, 1 seed (0), both arms | Each arm gets its own tuned rate. |
| Epochs and checkpoints | 3 epochs; validation every half epoch (6 checkpoints per run) | Ömer's rule 3. |
| Edge rule | If an arm's best rate is at a grid edge, run 1 more value beyond it (1e-5 or 1e-3) before choosing; at most once per edge | Ömer's rule 1. |
| Extension rule | If the selected checkpoint is the last one, resume that arm's run to 5 epochs before the seed runs | Ömer's rule 2. |
| Seeds | Sweep run is seed 0; the chosen configuration then runs with seeds 1 and 2 | 3 seeds per trained arm. |
| Modes | `model.train()` for steps, `model.eval()` for validation | Right after `get_peft_model`, all 24 LoRA dropout modules are in train mode (checked); the old code never switched modes. |

## Step 8: Where to train (14 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| Hardware | Modal, 1 A10G GPU per run (`src/finetune/modal_app.py`) | On the 16 GB Mac, arm C at batch 256 needed about 20 GB and swapped. With gradient checkpointing it fit (5.9 GB) but took about 4.3 s per step, about 13 h for the plan. |
| Volume contents | `train.pt`, `val.pt`, both negatives files, `captions_train2017.json`; no images and no test embeddings | Training needs nothing else; the test set stays off the training volume. |
| Git state | The local launcher sends its commit and dirty flag; runs refuse a dirty `src/` unless `--allow-dirty` (smoke runs only) | The container has no git repository. |

Step 8 smoke runs (16,384 images, 2 epochs, learning rate 1e-4, A10G; logs in `results/smoke/`):

| Arm | Time per step | Validation | Train loss (first → last checkpoint) |
|---|---|---|---|
| C `hard` | 0.383 s | 12.8 s | 1.19 → 0.86 |
| B `inbatch` | 0.216 s | 12.9 s | 0.91 → 0.71 |

Estimate for 1 full 3-epoch run (441 steps per epoch, 6 validations, plus about 1.5 min start-up): arm C about 11 min, arm B about 7.5 min. Modal price checked on 14 Nov 2026: A10 $0.000306 per second (about $1.10 per hour).

Approved by Ömer on 14 Nov 2026: worst case about 2.4 GPU-hours, about $2.70.

## Step 9: Sweep outcome, seed 0 (14 Nov 2026)

Logs: `results/runs/*_seed0.json`. Every checkpoint passed the R@1 guard (lowest R@1 0.345 against a guard of 0.290), so compositional accuracy decided.

| Arm | Grid result (best checkpoint per learning rate: comp. acc. / R@1) | Rule actions | Chosen |
|---|---|---|---|
| C `hard` | 2e-5: 0.961 / 0.387 (3.0); 1e-4: 0.974 / 0.401 (3.0); 5e-4: 0.979 / 0.402 (2.5); 1e-3 (edge run): 0.973 / 0.386 (1.0) | Edge rule ran 1e-3 because 5e-4 was the upper edge; 1e-3 did not win. Best checkpoint is not the last one, so no extension. | lr 5e-4, 3 epochs |
| B `inbatch` | 2e-5: 0.685 / 0.356 (0.5); 1e-4: 0.690 / 0.384 (0.5); 5e-4: 0.676 / 0.406 (2.5) | None: 1e-4 is inside the grid and its best checkpoint is the first one. | lr 1e-4, 3 epochs |

Notes:
- All arm B checkpoints are below base CLIP in validation compositional accuracy (0.652–0.690 against 0.721), so the rule picks its earliest checkpoint.
- The top arm C checkpoints differ by about 0.2 points, about 1 standard error on 4,986 examples.
- The epoch for each arm is chosen in step 10 from the seed-averaged metrics, with the same rule.

## Step 10: Final selection (14 Nov 2026)

`results/selection.json` (commit `908f5d2`, written at `6d8d443` with a clean `src/`): arm C lr 5e-4 epoch 2.5, arm B lr 1e-4 epoch 0.5, seeds 0–2. Seed-averaged validation: arm C compositional accuracy 0.979, R@1 0.403; arm B 0.685, 0.384. No extension: neither choice is the last checkpoint.

## Step 11: Test protocol (fixed before the test run, 14 Nov 2026)

| Item | Choice |
|---|---|
| Models | Base CLIP, plus the 3 selected seeds of arms B and C (7 models). Adapters checked: the downloaded seed-0 adapters reproduce their logged validation metrics exactly. |
| SugarCrepe | All 7 files (7,511 examples). Correct when cos(image, caption) > cos(image, negative); ties count as wrong. Accuracy per category and macro average over the 7 categories. |
| Retrieval | val2017, 5,000 images, 25,014 captions. Text-to-image: rank of the true image per caption. Image-to-text: rank of the best-ranked true caption per image. Ties count against the true match. R@1/5/10. |
| Table 1 | Mean and sample standard deviation over the 3 seeds per trained arm; base CLIP has 1 run. |
| Table 2 | Paired differences B − base, C − base, C − B. Per-example values averaged over the 3 seeds, then a cluster bootstrap that resamples image IDs (10,000 resamples, seed 0, 95% percentile interval). SugarCrepe categories and their macro average share each resample. |
| Run once | `src/eval/final_test.py` refuses to run if `results/test_results.json` exists, if `results/selection.json` is not committed, or if `src/` is dirty. Per-example outputs go to `results/test_results.npz`. |

## Step 12: Publish (14 Nov 2026)

| Decision | Choice | Reason |
|---|---|---|
| README | Claude drafts from the committed result files; Ömer edits | All numbers come from `results/`. The first version's numbers are removed, not quoted. |
| Demo | Uses the arm C seed-0 adapter (committed to `checkpoints/hard_lr5e-04_seed0_epoch2.5`, same SHA-256 as in `results/test_results.json`) and the 5,000 val2017 images as gallery | The old demo used `checkpoints/lora_epoch5`, trained on val2017. |
| Removed | `checkpoints/lora_epoch5`, `notebooks/` | Leaky setup or broken scripts; git history keeps them. |
| Dependencies | `requirements.txt` with exact pins of the direct dependencies, checked by a clean-clone test | Readable; the test shows it is sufficient. |

## Open decisions

- None.
- Where to cache the image embeddings and where to train (after the step 8 timing)
