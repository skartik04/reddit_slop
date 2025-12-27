#!/bin/bash
# ARC-AGI Quick Evaluation - All Your Models
# Tests abstract visual reasoning abilities

set -e

echo "============================================"
echo "ARC-AGI REASONING BENCHMARK"
echo "All models: Base + Fifth (5 sizes) + Benign (5 sizes)"
echo "============================================"
echo ""

# Start with evaluation set (400 tasks)
# Use --limit 50 for quick testing, remove for full evaluation

echo "Running ARC-AGI evaluation..."
echo "This will spawn parallel GPU jobs for each model"
echo ""

python run_eval_arc.py --generate \
  --model_configs base fifth_8 fifth_16 fifth_32 fifth_64 fifth_128 benign_8 benign_16 benign_32 benign_64 benign_128 \
  --dataset_splits evaluation \
  --limit 200

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo ""

# Analyze results
echo "Analyzing results..."
python run_eval_arc.py --evaluate

echo ""
echo "============================================"
echo "ARC-AGI EVALUATION COMPLETE!"
echo "============================================"
echo ""
echo "Results in: eval_results_arc/"
echo ""
echo "To run full evaluation (all 400 tasks), remove --limit flag"
echo ""
