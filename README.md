# Reddit Slop

Code and report for **Investigating "Intelligent Slop": A Small Case Study on How Reddit Content Affects an LLM**.

The final report is [reports/reddit-slop.pdf](reports/reddit-slop.pdf). Treat that PDF as the source of truth for the writeup and numbers.

## What This Project Does

This project fine-tunes `meta-llama/Llama-3.1-8B-Instruct` with LoRA on two small Reddit-derived datasets:

- `r/fifthworldproblems`: absurdist, structured, problem/solution-style posts.
- `r/benignexistence`: mundane, wholesome slice-of-life posts.

The report compares the resulting adapters against the base model on reasoning, factuality, creativity, and personality-style benchmarks. The main reported takeaway is mixed and content-dependent behavior: Fifth World fine-tuning showed consistent GSM8K gains in the tested low-data setting, while other benchmarks often degraded and trait scores shifted.

## Repository Layout

```text
reports/
  reddit-slop.pdf                  Final report
scripts/
  data/                            Reddit scraping and SFT data conversion
  train/                           LoRA training scripts
  eval/                            lm-eval, TRAIT, CounterFact, CS4, ARC helpers
  inference/                       Manual inference and AdvBench completion scripts
  setup_external_repos.sh          Clones external benchmark repos into external/
src/reddit_slop/
  paths.py                         Shared repo-relative path configuration
notebooks/                         Exploratory notebooks used during the project
artifacts/                         Local data, checkpoints, generated results; ignored by git
external/                          Local clones of CS4 and ARC-AGI; ignored by git
```

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `HF_TOKEN` in `.env` if you need gated Hugging Face model access.

Optional external benchmark repos:

```bash
scripts/setup_external_repos.sh
```

The code uses repo-relative defaults. Override paths only if needed:

```bash
export REDDIT_SLOP_BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct
export REDDIT_SLOP_ARTIFACTS_DIR=artifacts
export REDDIT_SLOP_HF_CACHE_DIR=.hf_cache
```

## Data

Raw and processed data live under `artifacts/data/` and are intentionally ignored by git.

Convert raw JSON data into chat-format JSONL:

```bash
python scripts/data/prepare_data.py
```

Collect new `r/benignexistence` data:

```bash
python scripts/data/collect_reddit.py
```

Expected processed files:

```text
artifacts/data/fifth_world_sft.jsonl
artifacts/data/benign_existence_sft.jsonl
```

## Training

Train Fifth World adapters:

```bash
python scripts/train/train_fifth_world.py --n 8
python scripts/train/train_fifth_world.py --n 16
python scripts/train/train_fifth_world.py --n 32
python scripts/train/train_fifth_world.py --n 64
```

Train Benign Existence adapters:

```bash
python scripts/train/train_benign.py --n 8
python scripts/train/train_benign.py --n 16
python scripts/train/train_benign.py --n 32
python scripts/train/train_benign.py --n 64
```

Adapters are written to `artifacts/checkpoints/`.

## Evaluation

Run standard lm-eval benchmarks:

```bash
python scripts/eval/eval_fifth_world.py --n 16 --benchmark gsm8k
python scripts/eval/eval_benign.py --n 16 --benchmark gsm8k
```

Run TRAIT and CounterFact:

```bash
python scripts/eval/eval_trait.py --type fifth --n 32
python scripts/eval/eval_counterfact.py --type benign --n 32
```

Run ARC-AGI and CS4 helpers:

```bash
scripts/eval/run_arc_quick.sh
scripts/eval/run_cs4_quick.sh
```

Evaluation outputs are written under `artifacts/eval_results*`.

## Inference

Compare base and LoRA completions manually:

```bash
python scripts/inference/compare_inference.py
```

Generate AdvBench completions:

```bash
python scripts/inference/advbench_completions.py --type fifth --n 16
python scripts/inference/advbench_completions.py --base_only
```

## Notes

- `artifacts/` is local-only and may contain large checkpoints, generated CSVs, logs, plots, and intermediate data.
- `external/` is local-only and contains copied/cloned benchmark repositories.
- The tracked repo is intended to contain the report, runnable scripts, notebooks, and lightweight metadata only.
