"""
Run all CS4 evaluations on the generated stories.

This script runs the following evaluations:
1. Constraint Satisfaction (requires OpenAI API)
2. Diversity (n-gram diversity)
3. Perplexity
4. Coherence vs Constraints
5. QUC and RCS scores

Usage:
    # Run all evaluations
    export OPENAI_API_KEY=your_key
    python run_cs4_evaluations.py --all
    
    # Run specific evaluations
    python run_cs4_evaluations.py --diversity --perplexity
"""

import argparse
import os
import subprocess
import sys
from glob import glob

RESULTS_DIR = '/mnt/SSD4/kartik/abstract/eval_results_cs4'
COMBINED_DIR = os.path.join(RESULTS_DIR, 'combined')
EVAL_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'evaluations')
CS4_EVAL_DIR = '/mnt/SSD4/kartik/abstract/cs4_benchmark/evaluation'


def check_openai_key():
    """Check if OpenAI API key is set."""
    if 'OPENAI_API_KEY' not in os.environ:
        print("❌ OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY=your_key")
        return False
    return True


def get_model_files():
    """Get all combined model CSV files."""
    csv_files = glob(os.path.join(COMBINED_DIR, '*.csv'))
    return sorted(csv_files)


def run_constraint_satisfaction():
    """Run constraint satisfaction evaluation using GPT-4."""
    print("\n" + "="*60)
    print("Running Constraint Satisfaction Evaluation")
    print("="*60 + "\n")
    
    if not check_openai_key():
        print("⚠️  Skipping constraint satisfaction (requires OpenAI API)")
        return
    
    os.makedirs(os.path.join(EVAL_OUTPUT_DIR, 'constraint_satisfaction'), exist_ok=True)
    
    model_files = get_model_files()
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.csv', '')
        output_file = os.path.join(EVAL_OUTPUT_DIR, 'constraint_satisfaction', f'{model_name}_evaluated.csv')
        
        print(f"Evaluating {model_name}...")
        
        cmd = [
            sys.executable,
            os.path.join(CS4_EVAL_DIR, 'constraint_satisfaction.py'),
            '--input_path', model_file,
            '--output_path', output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed: {e}\n")


def run_diversity_calculation():
    """Run diversity calculation (n-gram diversity)."""
    print("\n" + "="*60)
    print("Running Diversity Calculation")
    print("="*60 + "\n")
    
    os.makedirs(os.path.join(EVAL_OUTPUT_DIR, 'diversity'), exist_ok=True)
    
    model_files = get_model_files()
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.csv', '')
        output_file = os.path.join(EVAL_OUTPUT_DIR, 'diversity', f'{model_name}_diversity.csv')
        
        print(f"Calculating diversity for {model_name}...")
        
        cmd = [
            sys.executable,
            os.path.join(CS4_EVAL_DIR, 'diversity_calculation.py'),
            '--input_path', model_file,
            '--output_path', output_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed: {e}\n")


def generate_comparison_graphs():
    """Generate comparison graphs across all models."""
    print("\n" + "="*60)
    print("Generating Comparison Graphs")
    print("="*60 + "\n")
    
    os.makedirs(os.path.join(EVAL_OUTPUT_DIR, 'graphs'), exist_ok=True)
    
    model_files = get_model_files()
    
    if len(model_files) < 2:
        print("⚠️  Need at least 2 models for comparison graphs")
        return
    
    # Select up to 3 models for comparison
    # Priority: base_model, fifth_n8, benign_n8 (or first available)
    models_to_compare = []
    labels = []
    
    # Try to find base model
    base_files = [f for f in model_files if 'base_model' in os.path.basename(f)]
    if base_files:
        models_to_compare.append(base_files[0])
        labels.append('Base Model')
    
    # Try to find fifth model
    fifth_files = sorted([f for f in model_files if 'fifth' in os.path.basename(f)])
    if fifth_files:
        models_to_compare.append(fifth_files[0])
        model_name = os.path.basename(fifth_files[0]).replace('.csv', '')
        labels.append(f'Fifth ({model_name.split("_")[-1]})')
    
    # Try to find benign model
    benign_files = sorted([f for f in model_files if 'benign' in os.path.basename(f)])
    if benign_files:
        models_to_compare.append(benign_files[0])
        model_name = os.path.basename(benign_files[0]).replace('.csv', '')
        labels.append(f'Benign ({model_name.split("_")[-1]})')
    
    if len(models_to_compare) < 2:
        # Fallback: just use first 3 available
        models_to_compare = model_files[:3]
        labels = [os.path.basename(f).replace('.csv', '') for f in models_to_compare]
    
    print(f"Comparing models:")
    for label, file in zip(labels, models_to_compare):
        print(f"  - {label}: {os.path.basename(file)}")
    print()
    
    # Constraint Satisfaction Graph
    try:
        print("Generating constraint satisfaction graph...")
        # First check if evaluated files exist
        eval_files = []
        for model_file in models_to_compare:
            model_name = os.path.basename(model_file).replace('.csv', '')
            eval_file = os.path.join(EVAL_OUTPUT_DIR, 'constraint_satisfaction', f'{model_name}_evaluated.csv')
            if os.path.exists(eval_file):
                eval_files.append(eval_file)
            else:
                print(f"  ⚠️  Evaluated file not found for {model_name}")
        
        if len(eval_files) >= 2:
            output_graph = os.path.join(EVAL_OUTPUT_DIR, 'graphs', 'constraint_satisfaction.png')
            cmd = [
                sys.executable,
                os.path.join(CS4_EVAL_DIR, 'constraint_satisfaction_graph_generation.py'),
                '--file1', eval_files[0],
                '--file2', eval_files[1],
            ]
            if len(eval_files) >= 3:
                cmd.extend(['--file3', eval_files[2]])
            cmd.extend(['--output_file_path', output_graph])
            
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_graph}\n")
        else:
            print(f"  ⚠️  Need at least 2 evaluated models\n")
    except Exception as e:
        print(f"  ❌ Failed: {e}\n")
    
    # Diversity Graph
    try:
        print("Generating diversity comparison graph...")
        # Check if diversity files exist
        div_files = []
        for model_file in models_to_compare:
            model_name = os.path.basename(model_file).replace('.csv', '')
            div_file = os.path.join(EVAL_OUTPUT_DIR, 'diversity', f'{model_name}_diversity.csv')
            if os.path.exists(div_file):
                div_files.append(div_file)
        
        if len(div_files) >= 2:
            output_graph = os.path.join(EVAL_OUTPUT_DIR, 'graphs', 'diversity.png')
            cmd = [
                sys.executable,
                os.path.join(CS4_EVAL_DIR, 'diversity_graphs.py'),
                '--file1', div_files[0],
                '--file2', div_files[1],
            ]
            if len(div_files) >= 3:
                cmd.extend(['--file3', div_files[2]])
            cmd.extend(['--output_path', output_graph])
            
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_graph}\n")
        else:
            print(f"  ⚠️  Need diversity results first\n")
    except Exception as e:
        print(f"  ❌ Failed: {e}\n")
    
    # Perplexity Graph
    try:
        print("Generating perplexity comparison graph...")
        # Check if perplexity files exist
        perp_files = []
        for model_file in models_to_compare:
            model_name = os.path.basename(model_file).replace('.csv', '')
            perp_file = os.path.join(EVAL_OUTPUT_DIR, 'perplexity', f'{model_name}_perplexity.csv')
            if os.path.exists(perp_file):
                perp_files.append(perp_file)
        
        if len(perp_files) >= 2:
            output_graph = os.path.join(EVAL_OUTPUT_DIR, 'graphs', 'perplexity.png')
            cmd = [
                sys.executable,
                os.path.join(CS4_EVAL_DIR, 'perplexity_graphs.py'),
                '--file1', perp_files[0],
                '--file2', perp_files[1],
                '--label1', labels[0],
                '--label2', labels[1],
            ]
            if len(perp_files) >= 3:
                cmd.extend(['--file3', perp_files[2], '--label3', labels[2]])
            cmd.extend(['--output_path', output_graph])
            
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_graph}\n")
        else:
            print(f"  ⚠️  Need perplexity results first\n")
    except Exception as e:
        print(f"  ❌ Failed: {e}\n")


def print_summary():
    """Print summary of results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60 + "\n")
    
    print(f"Results saved to: {EVAL_OUTPUT_DIR}\n")
    
    print("Generated outputs:")
    for root, dirs, files in os.walk(EVAL_OUTPUT_DIR):
        level = root.replace(EVAL_OUTPUT_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


def run_perplexity_calculation():
    """Run perplexity calculation on all models."""
    print("\n" + "="*60)
    print("Running Perplexity Calculation")
    print("="*60 + "\n")
    
    os.makedirs(os.path.join(EVAL_OUTPUT_DIR, 'perplexity'), exist_ok=True)
    
    model_files = get_model_files()
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.csv', '')
        output_file = os.path.join(EVAL_OUTPUT_DIR, 'perplexity', f'{model_name}_perplexity.csv')
        
        print(f"Calculating perplexity for {model_name}...")
        
        cmd = [
            sys.executable,
            'calculate_perplexity.py',
            '--input_path', model_file,
            '--output_path', output_file,
            '--model', 'gpt2'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✅ Saved to {output_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Run CS4 Evaluations')
    
    parser.add_argument('--all', action='store_true', help='Run all evaluations (WARNING: includes costly GPT-4 API calls)')
    parser.add_argument('--free', action='store_true', help='Run only FREE evaluations (diversity + perplexity, no API needed)')
    parser.add_argument('--constraint_satisfaction', action='store_true', 
                        help='Run constraint satisfaction (requires OpenAI API, costs money)')
    parser.add_argument('--diversity', action='store_true', 
                        help='Run diversity calculation (FREE)')
    parser.add_argument('--perplexity_calc', action='store_true',
                        help='Run perplexity calculation (FREE)')
    parser.add_argument('--graphs', action='store_true', 
                        help='Generate comparison graphs')
    
    args = parser.parse_args()
    
    # If no specific evaluation selected, default to free metrics
    if not any([args.all, args.free, args.constraint_satisfaction, args.diversity, args.perplexity_calc, args.graphs]):
        print("No evaluation specified. Use --free for free metrics or --all for everything.")
        print("Defaulting to --free (diversity + perplexity, no API cost)")
        args.free = True
    
    # If --free is selected, enable free metrics
    if args.free:
        args.diversity = True
        args.perplexity_calc = True
        args.graphs = True
    
    # Check if combined files exist
    model_files = get_model_files()
    if not model_files:
        print("❌ No combined model files found!")
        print("   Run: python combine_cs4_results.py first")
        return
    
    print(f"\nFound {len(model_files)} model outputs to evaluate:")
    for f in model_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Run evaluations
    if args.all or args.constraint_satisfaction:
        if check_openai_key():
            run_constraint_satisfaction()
        else:
            print("⚠️  Skipping constraint satisfaction (no OpenAI API key)")
    
    if args.all or args.free or args.diversity:
        run_diversity_calculation()
    
    if args.all or args.free or args.perplexity_calc:
        run_perplexity_calculation()
    
    if args.all or args.free or args.graphs:
        generate_comparison_graphs()
    
    print_summary()


if __name__ == '__main__':
    main()
