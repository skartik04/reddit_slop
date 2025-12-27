"""
Combine CS4 results from all models into single CSV files for comparison.
"""

import os
import pandas as pd
from glob import glob

RESULTS_DIR = '/mnt/SSD4/kartik/abstract/eval_results_cs4'
OUTPUT_DIR = '/mnt/SSD4/kartik/abstract/eval_results_cs4/combined'

def combine_model_results(model_dir):
    """Combine all constraint level CSVs for a single model."""
    csv_files = glob(os.path.join(model_dir, 'constraints_*.csv'))
    
    if not csv_files:
        print(f"  ⚠️  No CSV files found in {model_dir}")
        return None
    
    dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def main():
    print("Combining CS4 results...\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all model directories
    model_dirs = [
        d for d in glob(os.path.join(RESULTS_DIR, '*'))
        if os.path.isdir(d) and os.path.basename(d) not in ['temp', 'combined']
    ]
    
    if not model_dirs:
        print("❌ No model directories found!")
        return
    
    print(f"Found {len(model_dirs)} model outputs:\n")
    
    for model_dir in sorted(model_dirs):
        model_name = os.path.basename(model_dir)
        print(f"Processing {model_name}...")
        
        combined_df = combine_model_results(model_dir)
        
        if combined_df is not None:
            output_file = os.path.join(OUTPUT_DIR, f'{model_name}.csv')
            combined_df.to_csv(output_file, index=False)
            print(f"  ✅ Saved {len(combined_df)} stories to {output_file}")
        
        print()
    
    print("="*60)
    print("COMBINATION COMPLETE!")
    print("="*60)
    print(f"\nCombined files saved to: {OUTPUT_DIR}")
    print("\nNext: Run constraint satisfaction evaluation (requires OpenAI API)")


if __name__ == '__main__':
    main()
