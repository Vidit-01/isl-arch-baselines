# ISL arch.md baselines

Train the nine-family ISL recognition baselines (11 checkpoints) on the **8-word** isolated-sign subset.

Videos are **not** in this repo. The cloud script downloads them from Hugging Face: [`vidit031/isl-isolated-8words`](https://huggingface.co/datasets/vidit031/isl-isolated-8words) (56 clips: yes, no, hello, water, eat, go, help, please).

Architecture notes: [`models/baselines/arch.md`](models/baselines/arch.md).

## Cloud GPU (RunPod / Lambda / Colab)

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

That run: install deps → download the 8-word set → extract MediaPipe landmarks → train all models under the **few-shot protocol** (locked test, k-shot train) → re-eval that test split → write `baselines_report.md`.

Quick GPU check (3 epochs, skip RGB CNN):

```bash
SMOKE=1 bash scripts/run_pipeline_baselines.sh
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

1. **Lock a balanced test set first** (target 20 clips/word), identity-disjoint from train/val. Grouping key is `User00x` when present, else `sessionN` in the path, else the clip itself.
2. **Draw 6–7 training clips/word** from the leftover pool. If that pool is larger than k, run several draws (`--draws 3`) and report mean ± std. Same locked test every draw so model comparisons stay paired.
3. **Val** is 1 leftover clip/word (needed for the val→test gap).
4. **Metrics:** per-class accuracy, macro-F1 as the headline, Wilson CIs on accuracy, bootstrap CIs on macro-F1 / macro-acc, McNemar + paired bootstrap when comparing models. Log **val−test gap** — that is the overfitting signal under scarcity.

`--strict-protocol` exits if any class has fewer than 20 test clips.

**Current 8-word Hugging Face set has 7 clips/word.** You cannot fill train 6–7 **and** test 20. The trainer degrades (typically train 5 / val 1 / test 1 per word), prints warnings, and will not silently claim a 20-shot test. Add more cross-signer clips before treating numbers as a real few-shot SLR result. On this 56-clip pool extra draws train the same leftover clips — the pipeline defaults to `--draws 1`. Use `--draws 3` once the leftover pool is larger than k.

Read the comparison on **which model degrades most gracefully**, not peak point accuracy. Raw-pixel `cnn_bilstm` is expected to fall hardest; landmark / spectral inputs should hold up better.

```bash
# honest run on today's 56 clips (degraded test size, warnings in the report)
python models/baselines/train.py --data-dir ISL_DATASET --models all --draws 1

# real protocol once you have ≥ ~28 clips/word
python models/baselines/train.py --data-dir ISL_DATASET --draws 3 --train-shots 7 --test-per-class 20 --strict-protocol
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
python scripts/download_hf_dataset.py --8words --out ISL_DATASET
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30 --data-dir ISL_DATASET
python models/baselines/train.py --data-dir ISL_DATASET --models all --draws 1
python models/baselines/eval.py --data-dir ISL_DATASET --models all
```
