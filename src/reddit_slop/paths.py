"""Shared paths and environment helpers for the Reddit Slop experiments."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(
    os.getenv("REDDIT_SLOP_ROOT", Path(__file__).resolve().parents[2])
).resolve()

ARTIFACTS_DIR = Path(
    os.getenv("REDDIT_SLOP_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
).expanduser().resolve()
DATA_DIR = Path(os.getenv("REDDIT_SLOP_DATA_DIR", ARTIFACTS_DIR / "data")).expanduser().resolve()
CHECKPOINT_DIR = Path(
    os.getenv("REDDIT_SLOP_CHECKPOINT_DIR", ARTIFACTS_DIR / "checkpoints")
).expanduser().resolve()
EVAL_RESULTS_DIR = Path(
    os.getenv("REDDIT_SLOP_EVAL_RESULTS_DIR", ARTIFACTS_DIR / "eval_results")
).expanduser().resolve()
BENIGN_EVAL_RESULTS_DIR = Path(
    os.getenv("REDDIT_SLOP_BENIGN_EVAL_RESULTS_DIR", ARTIFACTS_DIR / "eval_results_benign")
).expanduser().resolve()
ARC_RESULTS_DIR = Path(
    os.getenv("REDDIT_SLOP_ARC_RESULTS_DIR", ARTIFACTS_DIR / "eval_results_arc")
).expanduser().resolve()
CS4_RESULTS_DIR = Path(
    os.getenv("REDDIT_SLOP_CS4_RESULTS_DIR", ARTIFACTS_DIR / "eval_results_cs4")
).expanduser().resolve()
COMPLETIONS_DIR = Path(
    os.getenv("REDDIT_SLOP_COMPLETIONS_DIR", ARTIFACTS_DIR / "completions")
).expanduser().resolve()
EXTERNAL_DIR = Path(os.getenv("REDDIT_SLOP_EXTERNAL_DIR", REPO_ROOT / "external")).expanduser().resolve()
CS4_BENCHMARK_DIR = Path(
    os.getenv("REDDIT_SLOP_CS4_DIR", EXTERNAL_DIR / "cs4_benchmark")
).expanduser().resolve()
ARC_AGI_DIR = Path(os.getenv("REDDIT_SLOP_ARC_AGI_DIR", EXTERNAL_DIR / "ARC-AGI")).expanduser().resolve()

HF_CACHE_DIR = Path(
    os.getenv("REDDIT_SLOP_HF_CACHE_DIR", os.getenv("HF_HOME", REPO_ROOT / ".hf_cache"))
).expanduser().resolve()
BASE_MODEL = os.getenv("REDDIT_SLOP_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def load_environment() -> None:
    """Load optional environment files without requiring python-dotenv at import time."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = os.getenv("REDDIT_SLOP_ENV_FILE")
    candidates = [Path(env_file).expanduser()] if env_file else []
    candidates.append(REPO_ROOT / ".env")

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""

    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
