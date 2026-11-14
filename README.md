# CLIP-CompSearch

Text-side LoRA fine-tuning of CLIP with generated hard negatives, so that image search respects word order and relations ("a dog behind the shoes" vs "shoes behind the dog"). Only the text encoder is adapted; the vision encoder is frozen and its embeddings are cached.

All numbers below are from a held-out test run: no test image was used for training or for any choice of hyperparameters.

## Results

**Setup.** Training on 113,019 COCO train2017 images; model selection on a separate 5,000-image slice of train2017; test on all 5,000 COCO val2017 images and all 7,511 SugarCrepe examples. Three seeds per trained arm. Base CLIP is deterministic (1 run).

- **Arm B:** LoRA fine-tuning with in-batch negatives only.
- **Arm C:** the same, plus one generated hard negative per caption.

The two arms use identical images, captions, batches, steps and hyperparameter search; only the extra negatives differ. So **C − B measures the effect of the hard negatives**, and B − base measures the effect of fine-tuning on COCO captions.

### Table 1: Test accuracy (%; mean ± std over 3 seeds)

| SugarCrepe category (examples) | Base CLIP | B: in-batch only | C: + hard negatives |
|---|---|---|---|
| swap_obj (245) | 61.2 | 61.9 ± 1.6 | 71.7 ± 1.6 |
| swap_att (666) | 64.1 | 69.0 ± 0.5 | 73.1 ± 1.0 |
| replace_obj (1,652) | 90.7 | 93.2 ± 0.1 | 92.9 ± 0.6 |
| replace_att (788) | 80.3 | 84.3 ± 0.6 | 85.5 ± 0.7 |
| replace_rel (1,406) | 69.4 | 69.0 ± 0.1 | 72.9 ± 0.4 |
| add_obj (2,062) | 77.0 | 88.2 ± 0.6 | 87.8 ± 1.1 |
| add_att (692) | 69.5 | 79.7 ± 0.2 | 78.2 ± 0.7 |
| **Macro average** | **73.2** | **77.9 ± 0.2** | **80.3 ± 0.3** |

| Retrieval (val2017, 5,000 images, 25,014 captions) | Base CLIP | B: in-batch only | C: + hard negatives |
|---|---|---|---|
| Text → image R@1 | 30.5 | 37.5 ± 0.0 | 39.5 ± 0.1 |
| Text → image R@5 | 54.8 | 64.0 ± 0.1 | 67.0 ± 0.1 |
| Text → image R@10 | 66.2 | 74.6 ± 0.1 | 77.4 ± 0.2 |
| Image → text R@1 | 50.0 | 56.2 ± 0.2 | 57.4 ± 0.3 |
| Image → text R@5 | 74.8 | 79.6 ± 0.4 | 80.9 ± 0.1 |
| Image → text R@10 | 83.1 | 87.2 ± 0.1 | 88.3 ± 0.1 |

### Table 2: Paired differences (points, 95% interval)

Per-example results are averaged over the 3 seeds; the interval comes from a paired bootstrap that resamples **image IDs** (10,000 resamples), because one image can carry several SugarCrepe examples and five captions.

| Metric | C − B (hard negatives) | C − base | B − base |
|---|---|---|---|
| SugarCrepe swap_obj | +9.8 [+4.2, +15.5] | +10.5 [+4.5, +16.7] | +0.7 [−5.2, +6.4] |
| SugarCrepe swap_att | +4.1 [+1.3, +6.7] | +9.0 [+5.4, +12.5] | +4.9 [+1.7, +8.2] |
| SugarCrepe replace_obj | −0.3 [−1.1, +0.5] | +2.2 [+1.1, +3.4] | +2.5 [+1.6, +3.5] |
| SugarCrepe replace_att | +1.2 [−0.4, +2.8] | +5.2 [+2.8, +7.7] | +4.0 [+1.6, +6.4] |
| SugarCrepe replace_rel | +3.9 [+2.2, +5.7] | +3.5 [+1.2, +5.8] | −0.5 [−2.6, +1.7] |
| SugarCrepe add_obj | −0.3 [−1.2, +0.6] | +10.8 [+9.2, +12.4] | +11.2 [+9.6, +12.7] |
| SugarCrepe add_att | −1.5 [−3.8, +0.7] | +8.7 [+5.6, +11.8] | +10.2 [+7.2, +13.2] |
| **SugarCrepe macro average** | **+2.4 [+1.4, +3.4]** | **+7.1 [+5.9, +8.3]** | **+4.7 [+3.5, +5.9]** |
| Text → image R@1 | +2.0 [+1.6, +2.3] | +9.0 [+8.4, +9.7] | +7.1 [+6.5, +7.6] |
| Text → image R@5 | +3.0 [+2.6, +3.4] | +12.2 [+11.5, +12.9] | +9.2 [+8.6, +9.8] |
| Text → image R@10 | +2.9 [+2.5, +3.2] | +11.2 [+10.5, +11.8] | +8.3 [+7.7, +8.9] |
| Image → text R@1 | +1.2 [+0.4, +2.1] | +7.4 [+6.3, +8.5] | +6.2 [+5.1, +7.3] |
| Image → text R@5 | +1.3 [+0.6, +2.1] | +6.1 [+5.2, +7.1] | +4.8 [+3.9, +5.7] |
| Image → text R@10 | +1.1 [+0.5, +1.7] | +5.2 [+4.3, +6.0] | +4.1 [+3.3, +4.9] |

### What the results show

- **Hard negatives add +2.4 points of SugarCrepe macro accuracy** over the same fine-tune without them. The gain is concentrated where the generator works: it only makes swap negatives, and swap_obj (+9.8) and swap_att (+4.1) are the categories where it helps. On the add and replace categories, C − B is near zero, except replace_rel (+3.9), which was not predicted in advance.
- **About two-thirds of the gain over base CLIP comes from fine-tuning on COCO captions alone** (B − base: +4.7 of the +7.1 macro points).
- **Retrieval is not traded away.** Both arms improve text-to-image R@1 on the 5,000 val2017 images (30.5 → 37.5 and 39.5). These are COCO-style captions, so this is an in-domain result.
- Each comparison has 14 intervals without a multiple-comparison correction; one of them can exclude zero by chance. The headline metrics (SugarCrepe macro average and text-to-image R@1) were fixed before the test run.

Machine-readable results: [`results/test_results.json`](results/test_results.json); per-example outputs: `results/test_results.npz`.

## What changed from the first version

The first version of this project reported SugarCrepe and Recall@K gains that were **not valid**, and they have been removed:

1. **Train/test overlap.** It trained on COCO val2017, which is where the SugarCrepe images and the Recall@K test images come from.
2. **Selection on the test set.** Learning rate and epochs were chosen by SugarCrepe scores.
3. **The training script could not reproduce the run.** It read image file names as captions, so no hard negative was ever formed.

This version rebuilds the pipeline with a held-out split, a leakage check, a validation slice for every choice, 3 seeds and paired intervals. Every decision and its reason is logged in [`docs/decisions.md`](docs/decisions.md).

## Method

### Data split and leakage check
- **Train:** 113,287 train2017 images (566,741 captions). **Validation:** 5,000 train2017 images, sampled with seed 0 (25,012 captions). **Test:** all 5,000 val2017 images (25,014 captions).
- The image-ID lists are committed in `data/splits/`. A test asserts that the three sets share no image and that all 1,560 SugarCrepe images are in the test set ([`results/leakage_report.json`](results/leakage_report.json)).

### Hard negatives
- **Generator** (spaCy `en_core_web_sm`): swap the first noun with the first later, different noun; fall back to adjectives. Compound nouns ("tennis player"), hyphenated words ("t-shirt") and "X of Y" phrases ("a group of kids") move as one unit; position nouns ("in front of") are skipped; "a"/"an" is corrected after the swap.
- **Filter:** reject swaps whose two units are joined by "and"/"or" or by a symmetric relation (next to, beside, near, by, alongside, across from, opposite), because those swaps keep the meaning.
- **Output:** 502,010 caption–negative pairs from 566,741 training captions (12,020 captions had no swap; 52,711 swaps were rejected by the filter); 113,019 of 113,287 training images keep at least one pair ([`results/negatives_report.json`](results/negatives_report.json)).

#### Why a rule filter replaces NLI

The first version filtered the swaps with an NLI model (`cross-encoder/nli-deberta-v3-small`) and removed every pair labelled as entailment. A check on 2,000 COCO train2017 captions showed two problems:

- **It removed real negatives.** The model labelled 1,033 of 1,984 swap negatives (52%) as entailment. Many of them change the meaning, for example "A man in a crown wears a red jacket and is on the phone." → "A crown in a man wears a red jacket and is on the phone." A swap keeps every word of the caption, and NLI models tend to predict entailment when the premise and the hypothesis share their words (McCoy et al., 2019).
- **It kept swaps that keep the meaning.** It labelled "A dog and a cat sit on a sofa." → "A cat and a dog sit on a sofa." as contradiction.

The NLI scores are still stored with each pair, but they are not used for filtering.

### Training
- **Loss:** symmetric InfoNCE (mean of image→text and text→image cross-entropy). Arm C appends the hard negatives as extra image→text columns. Each image appears once per batch: one caption per image is sampled per epoch.
- **Model selection** (validation slice only): keep checkpoints whose text→image R@1 is at most 1 point below base CLIP; among those, choose the highest accuracy on validation hard negatives. Learning-rate grid 2e-5, 1e-4, 5e-4 for both arms with seed 0, validated every half epoch; an extra run beyond a grid edge if the best rate is at the edge (arm C: 1e-3, which did not win). The epoch is chosen on the 3-seed average ([`results/selection.json`](results/selection.json)).

| Parameter | Value |
|---|---|
| Base model | `openai/clip-vit-base-patch32` |
| LoRA | r = 16, α = 32, dropout 0.1 on `q_proj`, `v_proj` of the text encoder |
| Trainable parameters | 393,216 (0.62% of the text encoder, 0.26% of CLIP) |
| Logit scale | Fixed at CLIP's learned value, 100 |
| Batch size | 256 images (arm C: 512 texts per step) |
| Optimizer | AdamW, weight decay 0.01, 50 warmup steps, then constant |
| Selected learning rate | Arm C: 5e-4; arm B: 1e-4 |
| Selected checkpoint | Arm C: epoch 2.5 of 3; arm B: epoch 0.5 of 3 |
| Seeds | 0, 1, 2 |
| Hardware | 1 NVIDIA A10G on Modal; about 9 minutes per arm C run |

## Reproduce

### Setup
```bash
git clone https://github.com/kizilkayaom/clip-compsearch.git
cd clip-compsearch
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```
Use a git clone, not a ZIP download: every report records the git commit it ran on, and the tests check this. Tested with Python 3.13.5 on macOS (Apple M4); training ran on Linux with CUDA through Modal.

### Data
```bash
mkdir -p data/coco data/sugarcrepe
wget http://images.cocodataset.org/zips/train2017.zip -O data/coco/train2017.zip   # about 18 GB
wget http://images.cocodataset.org/zips/val2017.zip -O data/coco/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O data/coco/annotations.zip
unzip -q data/coco/train2017.zip -d data/coco/ && unzip -q data/coco/val2017.zip -d data/coco/
unzip -q data/coco/annotations.zip -d data/coco/
git clone https://github.com/RAIVNLab/sugar-crepe.git /tmp/sugar-crepe
cp /tmp/sugar-crepe/data/*.json data/sugarcrepe/
```

### Pipeline
The committed files in `data/splits/` and `results/` let you skip any step; each step writes a report with the git commit it ran on.

```bash
A=data/coco/annotations
python -m src.data.split --annotations-dir $A                 # data/splits/, results/split_report.json
python -m src.data.leakage                                    # results/leakage_report.json
python -m src.data.build_negatives --annotations-dir $A       # data/negatives/ (about 1 h on an M4 Mac)
python -m src.data.cache_embeddings --train2017-dir data/coco/train2017 --val2017-dir data/coco/val2017
python -m src.eval.validation --annotations-dir $A            # results/val_baseline.json
```

**Training on Modal** (the volume needs only the cached embeddings, the negatives and the caption file):
```bash
V=clip-compsearch-train
modal volume create $V
modal volume put $V embeddings/clip-vit-base-patch32/train.pt /embeddings/clip-vit-base-patch32/train.pt
modal volume put $V embeddings/clip-vit-base-patch32/val.pt /embeddings/clip-vit-base-patch32/val.pt
modal volume put $V data/negatives/train.jsonl /data/negatives/train.jsonl
modal volume put $V data/negatives/val.jsonl /data/negatives/val.jsonl
modal volume put $V $A/captions_train2017.json /annotations/captions_train2017.json

modal run --detach src/finetune/modal_app.py::sweep           # 6 runs in parallel
python -m src.finetune.select --arm hard                      # applies the selection and sweep rules
modal run --detach src/finetune/modal_app.py::run --arm hard --lr 5e-4 --seed 1
python -m src.finetune.select --final                         # results/selection.json, after seeds 1 and 2 of both arms
```
To train locally instead, run `python -m src.finetune.train --arm hard --lr 5e-4 --annotations-dir $A --gradient-checkpointing` (needed on a 16 GB Mac).

**Test (once):** download the selected adapters from the volume into `runs/`, then run
```bash
python -m src.eval.final_test --annotations-dir $A            # results/test_results.json and .npz
```
The script refuses to run if the results already exist, if `results/selection.json` is not committed, or if `src/` has uncommitted changes.

## Demo
```bash
python demo/app.py
```
A **Search** tab ranks the 5,000 val2017 images for a text query with base CLIP and with the fine-tuned model (arm C, seed 0, committed in `checkpoints/`). A **Matching** tab shows SugarCrepe examples and whether each model scores the caption above its negative. It needs `data/coco/val2017/`, `data/sugarcrepe/` and the cached `test.pt` embeddings.

## Limitations
- The generator only swaps nouns and adjectives. The gains are concentrated in the swap categories; the add and replace categories are not improved by the hard negatives.
- The validation hard negatives come from the same generator as the training negatives. Arm C's validation accuracy (97.9%) is much higher than its SugarCrepe swap accuracy, and for arm B the selection rule chose the earliest checkpoint (epoch 0.5), because its accuracy on generated negatives fell during training.
- Singular/plural agreement is not repaired after a swap ("A plastic trays is presented …"); the grammar error rate of the generated negatives has not been measured yet.
- One backbone (ViT-B/32), 3 seeds, and COCO-style captions only. swap_obj has 245 examples, so its interval is wide.

## Project Structure
```
clip-compsearch/
├── src/
│   ├── features.py            # CLIP loading and the shared feature functions
│   ├── data/                  # split, leakage check, negatives, image-embedding cache
│   ├── finetune/              # data, loss, training, selection rules, Modal launcher
│   ├── eval/                  # validation metrics, Recall@K, bootstrap, one-time test
│   └── search/                # first-version search helpers (not used by the pipeline)
├── tests/                     # pytest suite
├── demo/app.py                # Gradio demo
├── checkpoints/               # selected arm C adapter (seed 0)
├── data/splits/               # committed image-ID lists
├── results/                   # reports, run logs, selection and test results
└── docs/decisions.md          # every decision and its reason
```

## References
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Yuksekgonul et al., [When and why vision-language models behave like bags-of-words, and what to do about it?](https://arxiv.org/abs/2210.01936)
- Hsieh et al., [SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality](https://arxiv.org/abs/2306.14610)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Lin et al., [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- McCoy et al., [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/abs/1902.01007)

## Acknowledgements
- [COCO Dataset](https://cocodataset.org/) — CC-BY 4.0
- [HuggingFace PEFT](https://github.com/huggingface/peft) — Apache 2.0
- [spaCy](https://spacy.io/) — MIT License
