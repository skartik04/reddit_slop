#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p external

if [ ! -d external/cs4_benchmark/.git ]; then
  git clone https://github.com/anirudhlakkaraju/cs4_benchmark.git external/cs4_benchmark
else
  echo "external/cs4_benchmark already exists"
fi

if [ ! -d external/ARC-AGI/.git ]; then
  git clone https://github.com/fchollet/ARC-AGI.git external/ARC-AGI
else
  echo "external/ARC-AGI already exists"
fi
