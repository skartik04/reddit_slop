#!/bin/bash
# Setup script for CS4 benchmark evaluation

echo "Setting up CS4 Benchmark..."
echo ""

# Install CS4 requirements
echo "Installing CS4 requirements..."
pip install -r cs4_benchmark/requirements.txt

# Download NLTK data (needed for diversity calculation)
echo ""
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review CS4_BENCHMARK_README.md for full instructions"
echo "2. Run: ./run_cs4_full_pipeline.sh to generate stories"
echo "3. Run: python run_cs4_evaluations.py --all to evaluate"
