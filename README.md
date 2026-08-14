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

That run: install deps → download the 8-word set → extract MediaPipe landmarks → train all models → re-eval the held-out test split → write a comparison table.

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

## Outputs

After a full run:

| Path | Contents |
|---|---|
| `models/_weights/<model>/model.pt` | Weights + meta |
| `models/_weights/<model>/history.json` | Train/val curves |
| `models/_weights/<model>/test_metrics.json` | Held-out test acc/loss |
| `models/_weights/baselines_report.md` | Comparison table |
| `models/_weights/baselines_comparison.csv` | Same table as CSV |

## Manual steps

```bash
python -m pip install -r models/requirements.txt
python scripts/download_hf_dataset.py --8words --out ISL_DATASET
python models/mediapipe_transformer/extract_landmarks.py --num-frames 30 --data-dir ISL_DATASET
python models/baselines/train.py --data-dir ISL_DATASET --models all
python models/baselines/eval.py --data-dir ISL_DATASET --models all
```
