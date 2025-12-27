#!/bin/bash
# CS4 Quick Evaluation - All Your Models (FREE metrics only)
# No API key needed! Uses only Diversity + Perplexity

set -e

echo "============================================"
echo "CS4 CREATIVITY BENCHMARK - QUICK MODE"
echo "All models: Base + Fifth (5 sizes) + Benign (5 sizes)"
echo "============================================"
echo ""

# Use fewer constraints for faster testing (or use all: 7 15 23 31 39)
CONSTRAINT_LEVELS="7 15 23 31"

# Generate stories for ALL your models in the order you want
echo "Generating stories for all models..."
echo "This will spawn parallel GPU jobs for each model config"
echo ""

python run_eval_cs4.py --generate \
  --model_configs benign_16 benign_32 benign_64 benign_128 \
  --constraint_levels $CONSTRAINT_LEVELS

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo ""

# Combine results
echo "Combining results..."
python combine_cs4_results.py

echo ""
echo "============================================"
echo "Running FREE evaluations (no API key needed)"
echo "============================================"
echo ""

# Run only free metrics
python run_cs4_evaluations.py --diversity --perplexity_calc --graphs

echo ""
echo "============================================"
echo "EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results in: eval_results_cs4/evaluations/"
echo ""
echo "Optional: Set OPENAI_API_KEY and run constraint satisfaction:"
echo "  export OPENAI_API_KEY=your_key"
echo "  python run_cs4_evaluations.py --constraint_satisfaction"
echo ""
