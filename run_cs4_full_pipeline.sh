#!/bin/bash
# CS4 Benchmark - Complete Pipeline
# This script runs the full CS4 evaluation pipeline for 3 models

set -e  # Exit on error

echo "============================================"
echo "CS4 CREATIVITY BENCHMARK - FULL PIPELINE"
echo "============================================"
echo ""

# Configuration
CONSTRAINT_LEVELS="7 15 23"  # Start with fewer constraints for faster testing
N_VALUES="8 16 32"  # Your trained model sizes

# Step 1: Generate stories for BASE model
echo "Step 1/4: Generating stories for BASE model..."
python run_eval_cs4.py --generate --base_only --constraint_levels $CONSTRAINT_LEVELS

# Step 2: Generate stories for FIFTH models
echo ""
echo "Step 2/4: Generating stories for FIFTH models..."
python run_eval_cs4.py --generate --type fifth --n $N_VALUES --constraint_levels $CONSTRAINT_LEVELS

# Step 3: Generate stories for BENIGN models
echo ""
echo "Step 3/4: Generating stories for BENIGN models..."
python run_eval_cs4.py --generate --type benign --n $N_VALUES --constraint_levels $CONSTRAINT_LEVELS

# Step 4: Combine and prepare for evaluation
echo ""
echo "Step 4/4: Combining results..."
python combine_cs4_results.py

echo ""
echo "============================================"
echo "GENERATION COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Set OPENAI_API_KEY for constraint satisfaction evaluation"
echo "2. Run: python run_cs4_evaluations.py"
echo ""
