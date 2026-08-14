"""Cloud one-shot: download 8-word ISL set + train/eval all arch.md baselines.

Run from a GPU VM or Colab after cloning the GitHub repo (or let this script clone):

  python scripts/run_pipeline_baselines.py
  python scripts/run_pipeline_baselines.py --smoke          # 3-epoch sanity check
  python scripts/run_pipeline_baselines.py --skip-clone     # already inside the repo
  python scripts/run_pipeline_baselines.py --models stgcn hwgat ctr_gcn

Env (optional): HF_TOKEN, REPO_URL, ISL_WORKDIR, HF_DATASET

Do not use WORKDIR — Lightning / some notebooks set that to a folder that is
not this git checkout.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_CANDIDATE = Path(__file__).resolve().parents[1]
TRAIN_PY = Path("models") / "baselines" / "train.py"
EXTRACT_PY = Path("models") / "mediapipe_transformer" / "extract_landmarks.py"
EVAL_PY = Path("models") / "baselines" / "eval.py"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def is_checkout(path: Path) -> bool:
    return (path / TRAIN_PY).is_file() and (path / EXTRACT_PY).is_file()


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
    ap.add_argument(
        "--workdir",
        default=os.getenv("ISL_WORKDIR", ""),
        help="Clone destination if this script is not already inside the repo. "
        "Ignored when models/baselines/train.py is next to this scripts/ folder. "
        "Do not pass Lightning's WORKDIR.",
    )
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


def resolve_repo_root(args: argparse.Namespace) -> Path:
    """Prefer the checkout that contains this file, never Lightning's WORKDIR."""
    if is_checkout(ROOT_CANDIDATE):
        return ROOT_CANDIDATE
    cwd = Path.cwd()
    if is_checkout(cwd):
        return cwd
    explicit = (args.workdir or "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if is_checkout(p):
            return p
        nested = p / "isl-arch-baselines"
        if is_checkout(nested):
            return nested
        return nested if p.name != "isl-arch-baselines" else p
    return Path(os.getenv("HOME", str(Path.cwd()))) / "isl-arch-baselines"


def resolve_data_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    raw = Path(args.data_dir).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        (repo_root / raw).resolve(),
        (Path.cwd() / raw).resolve(),
        Path("/home/zeus/content") / raw.name,
        Path.home() / "content" / raw.name,
    ]
    for path in candidates:
        if (path / "metadata.csv").exists():
            print(f"reusing dataset at {path}")
            return path
    return candidates[0]


def clone_if_needed(repo_url: str, repo_root: Path, skip: bool, branch: str = "") -> None:
    if is_checkout(repo_root):
        print(f"using existing checkout {repo_root}")
        return
    if skip:
        raise SystemExit(
            f"SKIP clone but {repo_root} is not an isl-arch-baselines checkout "
            f"(missing {repo_root / TRAIN_PY})"
        )
    print(f"Cloning {repo_url} -> {repo_root}")
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    branch = (branch or os.getenv("GIT_BRANCH", "")).strip()
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(repo_root)]
    _run(cmd)
    if not is_checkout(repo_root):
        raise SystemExit(f"clone finished but {repo_root / EXTRACT_PY} is missing")


def install_deps(repo_root: Path) -> None:
    py = sys.executable
    _run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    req_models = repo_root / "models" / "requirements.txt"
    req_root = repo_root / "requirements.txt"
    if req_models.exists():
        _run([py, "-m", "pip", "install", "-r", str(req_models)])
    if req_root.exists():
        _run([py, "-m", "pip", "install", "-r", str(req_root)])
    _run([py, "-m", "pip", "install", "-U", "huggingface_hub"])


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo_root = resolve_repo_root(args)
    clone_if_needed(args.repo_url, repo_root, args.skip_clone, branch=args.branch)
    os.chdir(repo_root)
    print(f"REPO_ROOT={repo_root}")

    extract = repo_root / EXTRACT_PY
    train = repo_root / TRAIN_PY
    if not extract.is_file() or not train.is_file():
        raise SystemExit(
            f"repo root is wrong: {repo_root}\n"
            f"  extract exists={extract.is_file()} ({extract})\n"
            f"  train exists={train.is_file()} ({train})\n"
            f"Lightning WORKDIR is ignored; run from the isl-arch-baselines clone."
        )

    if args.install_deps:
        install_deps(repo_root)

    py = sys.executable
    data_dir = resolve_data_dir(args, repo_root)
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
                str(extract),
                "--num-frames",
                str(args.num_frames),
                "--data-dir",
                str(data_dir),
            ],
            cwd=repo_root,
        )

    if not args.skip_train:
        print("=== Train arch.md baselines ===")
        cmd = [
            py,
            str(train),
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
        _run(cmd, cwd=repo_root)

        print("=== Re-eval held-out test split ===")
        eval_models = models if models != ["all"] else ["all"]
        _run(
            [
                py,
                str(repo_root / EVAL_PY),
                "--data-dir",
                str(data_dir),
                "--models",
                *eval_models,
                "--seed",
                str(args.seed),
                *(["--cpu"] if args.cpu else []),
            ],
            cwd=repo_root,
        )

    sys.path.insert(0, str(repo_root / "models"))
    from baselines.report import write_report  # noqa: WPS433

    write_report()
    print("=== PIPELINE COMPLETE ===")
    print(f"Weights + metrics: {repo_root / 'models' / '_weights'}")
    print(f"Comparison table:  {repo_root / 'models' / '_weights' / 'baselines_report.md'}")


if __name__ == "__main__":
    main()
