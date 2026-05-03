#!/bin/bash
# Setup script for CS4 benchmark evaluation

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "Setting up CS4 Benchmark..."
echo ""

# Install CS4 requirements
echo "Installing CS4 requirements..."
pip install -r external/cs4_benchmark/requirements.txt

# Download NLTK data (needed for diversity calculation)
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the README CS4 section for full instructions"
echo "2. Run: scripts/eval/run_cs4_full_pipeline.sh to generate stories"
echo "3. Run: python scripts/eval/eval_cs4_metrics.py --all to evaluate"
