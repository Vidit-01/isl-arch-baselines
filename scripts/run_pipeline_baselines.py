"""Cloud one-shot: download 8-word ISL set + train/eval all arch.md baselines.

Run from a GPU VM or Colab after cloning the GitHub repo (or let this script clone):

  python scripts/run_pipeline_baselines.py
  python scripts/run_pipeline_baselines.py --smoke          # 3-epoch sanity check
  python scripts/run_pipeline_baselines.py --skip-clone     # already inside the repo
  python scripts/run_pipeline_baselines.py --models stgcn hwgat ctr_gcn

Env (optional): HF_TOKEN, REPO_URL, WORKDIR, HF_DATASET
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_CANDIDATE = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _snapshot_download(repo: str, out: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(repo_id=repo, repo_type="dataset", local_dir=str(out), token=token)
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download 8-word ISL set and train arch.md baselines")
    ap.add_argument("--repo-url", default=os.getenv("REPO_URL", "https://github.com/Vidit-01/isl-arch-baselines.git"))
    ap.add_argument("--branch", default=os.getenv("GIT_BRANCH", "main"), help="Git branch to clone")
    ap.add_argument("--workdir", default=os.getenv("WORKDIR", ""), help="Clone destination; default = this repo")
    ap.add_argument("--hf-dataset", default=os.getenv("HF_DATASET", "vidit031/isl-isolated-8words"))
    ap.add_argument("--data-dir", default="ISL_DATASET", help="Where to put the 8-word clips")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-clone", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-landmarks", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, skip RGB CNN (quick GPU check)")
    ap.add_argument("--install-deps", action="store_true", default=True)
    ap.add_argument("--no-install-deps", action="store_false", dest="install_deps")
    return ap.parse_args()


def resolve_workdir(args: argparse.Namespace) -> Path:
    if args.workdir:
        return Path(args.workdir).expanduser().resolve()
    if args.skip_clone:
        if (Path.cwd() / "models" / "baselines" / "train.py").exists():
            return Path.cwd()
        return ROOT_CANDIDATE
    if (ROOT_CANDIDATE / "models" / "baselines" / "train.py").exists():
        return ROOT_CANDIDATE
    return Path(os.getenv("HOME", str(Path.cwd()))) / "isl-arch-baselines"


def clone_if_needed(repo_url: str, workdir: Path, skip: bool, branch: str = "") -> None:
    if skip:
        print(f"SKIP clone; using {workdir}")
        return
    if (workdir / ".git").exists():
        print(f"Updating {workdir}")
        _run(["git", "fetch", "--all", "--prune"], cwd=workdir)
        try:
            _run(["git", "pull", "--ff-only"], cwd=workdir)
        except subprocess.CalledProcessError:
            print("git pull --ff-only failed; continuing with local tree")
        return
    if (workdir / "models" / "baselines" / "train.py").exists():
        print(f"Repo files already present at {workdir}")
        return
    print(f"Cloning {repo_url} -> {workdir}")
    workdir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    branch = (branch or os.getenv("GIT_BRANCH", "")).strip()
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(workdir)]
    _run(cmd)


def install_deps(workdir: Path) -> None:
    py = sys.executable
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    req_models = workdir / "models" / "requirements.txt"
    req_root = workdir / "requirements.txt"
    if req_models.exists():
        _run([py, "-m", "pip", "install", "-r", str(req_models)])
    if req_root.exists():
        _run([py, "-m", "pip", "install", "-r", str(req_root)])
    _run([py, "-m", "pip", "install", "-U", "huggingface_hub"])


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    workdir = resolve_workdir(args)
    clone_if_needed(args.repo_url, workdir, args.skip_clone, branch=args.branch)
    os.chdir(workdir)
    print(f"WORKDIR={workdir}")

    if args.install_deps:
        install_deps(workdir)

    py = sys.executable
    data_dir = (workdir / args.data_dir).resolve()
    token = os.getenv("HF_TOKEN")

    if not args.skip_download:
        print(f"=== Hugging Face dataset {args.hf_dataset} -> {data_dir} ===")
        _snapshot_download(args.hf_dataset, data_dir, token)
    meta = data_dir / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"missing {meta} — download failed or --data-dir is wrong")
    n_mp4 = sum(1 for _ in data_dir.rglob("*.mp4"))
    print(f"mp4 count={n_mp4}  metadata={meta.exists()}")
    if n_mp4 < 8:
        raise SystemExit(f"too few videos ({n_mp4}). Check HF LFS / HF_TOKEN.")

    models = list(args.models)
    epochs = args.epochs
    if args.smoke:
        epochs = epochs or 3
        if models == ["all"]:
            # RGB CNN is slow to download/init; skip on a 3-epoch probe
            models = [
                "mp_bilstm",
                "mp_transformer",
                "stgcn",
                "ctr_gcn",
                "td_gcn",
                "hwgat",
                "fft_bilstm",
                "cwt_bilstm",
                "cwt_transformer",
                "pgf_slr",
            ]

    workers = args.num_workers
    if workers is None:
        workers = 0 if sys.platform == "win32" else 4

    if not args.skip_landmarks:
        print(f"=== MediaPipe landmarks T={args.num_frames} ===")
        _run(
            [
                py,
                "models/mediapipe_transformer/extract_landmarks.py",
                "--num-frames",
                str(args.num_frames),
                "--data-dir",
                str(data_dir),
            ],
            cwd=workdir,
        )

    if not args.skip_train:
        print("=== Train arch.md baselines ===")
        cmd = [
            py,
            "models/baselines/train.py",
            "--data-dir",
            str(data_dir),
            "--models",
            *models,
            "--num-frames",
            str(args.num_frames),
            "--num-workers",
            str(workers),
            "--seed",
            str(args.seed),
        ]
        if epochs is not None:
            cmd.extend(["--epochs", str(epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.cpu:
            cmd.append("--cpu")
        _run(cmd, cwd=workdir)

        print("=== Re-eval held-out test split ===")
        eval_models = models if models != ["all"] else ["all"]
        _run(
            [
                py,
                "models/baselines/eval.py",
                "--data-dir",
                str(data_dir),
                "--models",
                *eval_models,
                "--seed",
                str(args.seed),
                *(["--cpu"] if args.cpu else []),
            ],
            cwd=workdir,
        )

    sys.path.insert(0, str(workdir / "models"))
    from baselines.report import write_report  # noqa: WPS433

    write_report()
    print("=== PIPELINE COMPLETE ===")
    print(f"Weights + metrics: {workdir / 'models' / '_weights'}")
    print(f"Comparison table:  {workdir / 'models' / '_weights' / 'baselines_report.md'}")


if __name__ == "__main__":
    main()
