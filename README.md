# ISL arch.md baselines

Train the nine-family ISL recognition baselines (11 checkpoints) on isolated ISL.

Videos are **not** in this repo. The cloud script downloads [`vidit031/isl-isolated-40words`](https://huggingface.co/datasets/vidit031/isl-isolated-40words) (642 clips, 40 glosses) and **keeps the 8 classes with the most videos** — not a fixed random 8-word list.

Architecture notes: [`models/baselines/arch.md`](models/baselines/arch.md).

## Cloud GPU (Lightning / RunPod / Lambda / Colab)

The pipeline downloads the **40-word** Hub set, ignores any leftover 56-clip `ISL_DATASET`, picks the 8 highest-count glosses, and trains the few-shot protocol (7-shot train, 15-clip test, 3 draws).

**Lightning AI** (existing clone):

```bash
cd /teamspace/studios/this_studio/isl-arch-baselines
git pull origin main
python scripts/run_pipeline_baselines.py --skip-clone
```

First run will download ~642 clips into `/home/zeus/content/ISL_DATASET_40WORDS` (not the old `ISL_DATASET`), extract MediaPipe landmarks, then train. Do **not** pass `--skip-download` or `--data-dir /home/zeus/content/ISL_DATASET` — that folder is the small 8-word copy.

Fresh VM:

```bash
git clone https://github.com/Vidit-01/isl-arch-baselines.git
cd isl-arch-baselines
bash scripts/run_pipeline_baselines.sh
```

Colab:

```python
!git clone https://github.com/Vidit-01/isl-arch-baselines.git
%cd isl-arch-baselines
!python scripts/run_pipeline_baselines.py --skip-clone
```

Quick GPU check (3 epochs, skip RGB CNN, 1 draw):

```bash
python scripts/run_pipeline_baselines.py --skip-clone --smoke
```

If the Hugging Face dataset is private: `export HF_TOKEN=hf_...`

## What gets trained

| CLI name | Model |
|---|---|
| `cnn_bilstm` | CNN + BiLSTM (RGB) |
| `mp_bilstm` | MediaPipe + BiLSTM |
| `mp_transformer` | MediaPipe + Transformer |
| `stgcn` | ST-GCN |
| `ctr_gcn` / `td_gcn` | CTR-GCN / TD-GCN |
| `hwgat` | HWGAT |
| `fft_bilstm` | FFT + kinematic + BiLSTM |
| `cwt_bilstm` / `cwt_transformer` | CWT + BiLSTM / Transformer |
| `pgf_slr` | Graph-Fourier attention (PGF-SLR-style) |

Subset:

```bash
python scripts/run_pipeline_baselines.py --skip-clone --models stgcn hwgat ctr_gcn
```

## Few-shot protocol

Default split is **not** a random 70/15/15. `train.py --protocol fewshot` (the default):

1. **Lock a balanced test set first** (target 15 clips/word), identity-disjoint from train/val. Grouping key is `User00x` when present, else `sessionN` in the path, else the clip itself.
2. **Draw 6–7 training clips/word** from the leftover pool. If that pool is larger than k, run several draws (`--draws 3`) and report mean ± std. Same locked test every draw so model comparisons stay paired.
3. **Val** is 1 leftover clip/word (needed for the val→test gap).
4. **Metrics:** per-class accuracy, macro-F1 as the headline, Wilson CIs on accuracy, bootstrap CIs on macro-F1 / macro-acc, McNemar + paired bootstrap when comparing models. Log **val−test gap** — that is the overfitting signal under scarcity.

`--strict-protocol` exits if any class has fewer than 15 test clips.

**Class selection:** default is the **8 glosses with the most clips** in whatever `--data-dir` you pass (`--n-words 8`, or `--words top8`). On `vidit031/isl-isolated-40words` that is currently `thank you` (39), `friend` (38), `school` (37), `hello` (36), `market` (36), `okay` (23), `hospital` (21), `sit` (19). Pass `--words all` for every class, `--words legacy8` for the old eat/go/hello/help/no/please/water/yes list, or name glosses (`thank_you` maps to `thank you`).

**Tiny 8-word smoke set** (`vidit031/isl-isolated-8words`, 7 clips/word) cannot fill train 6–7 **and** test 15. The trainer degrades and warns. Prefer the 40-word corpus. On that set, `hello` / `thank you` / `friend` / `school` / `market` reach a 15-shot test after reserving train 7 + val 1; `okay` is exact (23); `hospital` and `sit` still fall a few clips short. Pipeline `--draws` defaults to 1; use `--draws 3` when leftover > k.

Read the comparison on **which model degrades most gracefully**, not peak point accuracy. Raw-pixel `cnn_bilstm` is expected to fall hardest; landmark / spectral inputs should hold up better.

```bash
# 40-word corpus, 8 highest-count classes, 15-shot test
python scripts/download_hf_dataset.py --repo vidit031/isl-isolated-40words --out ISL_DATASET_40WORDS
python models/baselines/train.py --data-dir ISL_DATASET_40WORDS --n-words 8 --draws 3 --train-shots 7 --test-per-class 15

# old 8-word smoke list
python models/baselines/train.py --data-dir ISL_DATASET --words legacy8 --draws 1
```

Old random split: `--protocol stratified`.

## Outputs

After a full run:

| Path | Contents |
|---|---|
| `models/_weights/<model>/model.pt` | Weights + meta |
| `models/_weights/<model>/history.json` | Train/val curves |
| `models/_weights/<model>/test_metrics.json` | Test acc, macro-F1, Wilson/bootstrap CIs, per-class, preds |
| `models/_weights/fewshot_protocol.json` | Locked test paths + per-draw train/val |
| `models/_weights/baselines_report.md` | Headline table, per-class errors, McNemar |
| `models/_weights/baselines_pairwise.json` | McNemar + paired bootstrap |
| `models/_weights/draws/<dd>/<name>/` | Per-draw weights when `--draws > 1` |

## Manual steps

```bash
python -m pip install -r models/requirements.txt
python scripts/download_hf_dataset.py --repo vidit031/isl-isolated-40words --out ISL_DATASET
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30 --data-dir ISL_DATASET
python models/baselines/train.py --data-dir ISL_DATASET --models all --n-words 8 --draws 3
python models/baselines/eval.py --data-dir ISL_DATASET --models all
```
