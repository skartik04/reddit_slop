#!/bin/bash
# ARC-AGI Quick Evaluation - All Your Models
# Tests abstract visual reasoning abilities

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "============================================"
echo "ARC-AGI REASONING BENCHMARK"
echo "All report models: Base + Fifth (4 sizes) + Benign (4 sizes)"
echo "============================================"
echo ""

# Start with evaluation set (400 tasks)
# Use --limit 50 for quick testing, remove for full evaluation

echo "Running ARC-AGI evaluation..."
echo "This will spawn parallel GPU jobs for each model"
echo ""

python scripts/eval/eval_arc.py --generate \
  --model_configs base fifth_8 fifth_16 fifth_32 fifth_64 benign_8 benign_16 benign_32 benign_64 \
  --dataset_splits evaluation \
  --limit 200

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo ""

# Analyze results
echo "Analyzing results..."
python scripts/eval/eval_arc.py --evaluate

echo ""
echo "============================================"
echo "ARC-AGI EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results in: artifacts/eval_results_arc/"
echo ""
echo "To run full evaluation (all 400 tasks), remove --limit flag"
echo ""
