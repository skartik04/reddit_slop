#!/bin/bash
# CS4 Quick Evaluation - All Your Models (FREE metrics only)
# No API key needed! Uses only Diversity + Perplexity

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "============================================"
echo "CS4 CREATIVITY BENCHMARK - QUICK MODE"
echo "All report models: Base + Fifth (4 sizes) + Benign (4 sizes)"
echo "============================================"
echo ""

# Use fewer constraints for faster testing (or use all: 7 15 23 31 39)
CONSTRAINT_LEVELS="7 15 23 31"

# Generate stories for ALL your models in the order you want
echo "Generating stories for all models..."
echo "This will spawn parallel GPU jobs for each model config"
echo ""

python scripts/eval/eval_cs4.py --generate \
  --model_configs base fifth_8 fifth_16 fifth_32 fifth_64 benign_8 benign_16 benign_32 benign_64 \
  --constraint_levels $CONSTRAINT_LEVELS

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo ""

# Combine results
echo "Combining results..."
python scripts/eval/combine_cs4_results.py

echo ""
echo "============================================"
echo "Running FREE evaluations (no API key needed)"
echo "============================================"
echo ""

# Run only free metrics
python scripts/eval/eval_cs4_metrics.py --diversity --perplexity_calc --graphs

echo ""
echo "============================================"
echo "EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results in: artifacts/eval_results_cs4/evaluations/"
echo ""
echo "Optional: Set OPENAI_API_KEY and run constraint satisfaction:"
echo "  export OPENAI_API_KEY=your_key"
echo "  python scripts/eval/eval_cs4_metrics.py --constraint_satisfaction"
echo ""
