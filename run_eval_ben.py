"""
Run lm_eval benchmarks on Base Model and LoRA Model

Usage:
    python run_eval.py --n 8 --benchmark all
    python run_eval.py --n 8 --benchmark riddlesense
    python run_eval.py --n 8 --benchmark arc_challenge --skip_base
    python run_eval.py --n 8 --benchmark bbh_all --num_fewshot 5
"""

import argparse
import subprocess
import os
import json
from datetime import datetime

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = '/mnt/SSD4/kartik/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_CHECKPOINT_DIR = '/mnt/SSD4/kartik/abstract/checkpoints'
RESULTS_DIR = '/mnt/SSD4/kartik/abstract/eval_results_benign'
DEFAULT_NUM_FEWSHOT = 3

# Benchmarks that need limiting (too many questions)
LIMIT_BENCHMARKS = {'gsm8k': 250, 'hellaswag': 250}

# Benchmark task mappings (exact lm_eval task names)
BENCHMARKS = {
    # BBH tasks (cot_fewshot = chain-of-thought with built-in few-shot)
    'bbh_reasoning': 'bbh_cot_fewshot_reasoning_about_colored_objects',
    'bbh_navigate': 'bbh_cot_fewshot_navigate',
    'bbh_geometric': 'bbh_cot_fewshot_geometric_shapes',
    'bbh_logical': 'bbh_cot_fewshot_logical_deduction_seven_objects',
    
    # Other benchmarks
    'riddlesense': 'bigbench_riddle_sense_multiple_choice',
    'arc_challenge': 'arc_challenge',
    'gsm8k': 'gsm8k',
    'hellaswag': 'hellaswag',
    
    # Multiple BBH at once
    'bbh_all': 'bbh_cot_fewshot_reasoning_about_colored_objects,bbh_cot_fewshot_navigate,bbh_cot_fewshot_geometric_shapes,bbh_cot_fewshot_logical_deduction_seven_objects',
    'reqd': 'riddlesense,arc_challenge,gsm8k,hellaswag',
    # ALL benchmarks at once
    'all': 'bbh_cot_fewshot_reasoning_about_colored_objects,bbh_cot_fewshot_navigate,bbh_cot_fewshot_geometric_shapes,bbh_cot_fewshot_logical_deduction_seven_objects,bigbench_riddle_sense_multiple_choice,arc_challenge,gsm8k,hellaswag',
}

def run_lm_eval(model_path, task, output_path, num_fewshot=3, device='cuda:0', limit=None, is_lora=False, base_model_path=None):
    """Run lm_eval on a model (exactly like CLI)."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if is_lora:
        # For LoRA: pretrained=base,peft=adapter
        model_args = f"pretrained={base_model_path},peft={model_path}"
    else:
        model_args = f"pretrained={model_path}"
    
    # Exact same format as CLI command
    cmd = [
        'lm_eval',
        '--model', 'hf',
        '--model_args', model_args,
        '--tasks', task,
        '--device', device,
        '--batch_size', 'auto',
        '--num_fewshot', str(num_fewshot),
        '--output_path', output_path,
        '--seed', '42',  # Fixed seed for reproducibility
    ]
    
    # Add limit if specified (for large benchmarks like gsm8k, hellaswag)
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    print(f"\n{'='*60}")
    print(f"Running lm_eval...")
    print(f"Model: {model_path}")
    print(f"Tasks: {task}")
    print(f"Few-shot: {num_fewshot}")
    if limit:
        print(f"Limit: {limit} samples")
    print(f"Output: {output_path}")
    if is_lora:
        print(f"LoRA adapter on base: {base_model_path}")
    print(f"{'='*60}\n")
    
    # Print full command for debugging
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def load_results(results_dir):
    """Load results from the output directory."""
    # Find the results.json file
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith('results_') and f.endswith('.json'):
                path = os.path.join(root, f)
                with open(path) as file:
                    return json.load(file)
    return None

def print_comparison(base_results, lora_results, n):
    """Print a nice comparison of results."""
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"                    BENCHMARK RESULTS COMPARISON")
    print(f"                      Base Model vs LoRA (n={n})")
    print(f"{'='*70}")
    print(f"{'='*70}")
    
    if not base_results or not lora_results:
        print("Could not load results!")
        return
    
    base_scores = base_results.get('results', {})
    lora_scores = lora_results.get('results', {})
    
    print(f"\n{'Task':<50} {'Base':>10} {'LoRA':>10} {'Delta':>10}")
    print(f"{'-'*80}")
    
    for task in base_scores:
        base_task = base_scores.get(task, {})
        lora_task = lora_scores.get(task, {})
        
        # Find the main metric (usually acc or exact_match)
        for metric in ['acc,none', 'acc_norm,none', 'exact_match,get-answer', 'acc']:
            if metric in base_task:
                base_val = base_task[metric]
                lora_val = lora_task.get(metric, 0)
                delta = lora_val - base_val
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                print(f"{task:<50} {base_val:>10.4f} {lora_val:>10.4f} {delta_str:>10}")
                break
    
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Run lm_eval benchmarks on Base and LoRA models')
    parser.add_argument('--n', type=int, required=True, help='LoRA checkpoint N (e.g., 8, 16, 32)')
    parser.add_argument('--benchmark', type=str, required=True, 
                        choices=list(BENCHMARKS.keys()),
                        help=f'Benchmark to run: {list(BENCHMARKS.keys())}')
    parser.add_argument('--num_fewshot', type=int, default=DEFAULT_NUM_FEWSHOT, 
                        help=f'Number of few-shot examples (default: {DEFAULT_NUM_FEWSHOT})')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--skip_base', action='store_true', help='Skip base model evaluation (only run LoRA)')
    parser.add_argument('--skip_lora', action='store_true', help='Skip LoRA model evaluation (only run base)')
    
    args = parser.parse_args()
    
    # Paths
    lora_path = f'{LORA_CHECKPOINT_DIR}/benign_lora_n{args.n}'
    task = BENCHMARKS[args.benchmark]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if this benchmark needs limiting
    limit = LIMIT_BENCHMARKS.get(args.benchmark, None)
    
    # Check LoRA exists
    if not os.path.exists(lora_path):
        print(f"❌ LoRA checkpoint not found: {lora_path}")
        print(f"Available checkpoints:")
        if os.path.exists(LORA_CHECKPOINT_DIR):
            for d in os.listdir(LORA_CHECKPOINT_DIR):
                print(f"  - {d}")
        return
    
    # Output paths
    base_output = f'{RESULTS_DIR}/{args.benchmark}/base_{timestamp}'
    lora_output = f'{RESULTS_DIR}/{args.benchmark}/lora_n{args.n}_{timestamp}'
    
    print(f"\n{'#'*70}")
    print(f"EVALUATION: {args.benchmark}")
    print(f"LoRA: n={args.n}")
    print(f"Device: {args.device}")
    print(f"Few-shot: {args.num_fewshot}")
    if limit:
        print(f"Limit: {limit} samples (seed=42)")
    print(f"Run base: {not args.skip_base}")
    print(f"Run LoRA: {not args.skip_lora}")
    print(f"{'#'*70}")
    
    # Run base model eval
    if not args.skip_base:
        print("\n🔵 Evaluating BASE MODEL...")
        success = run_lm_eval(BASE_MODEL_PATH, task, base_output, num_fewshot=args.num_fewshot, device=args.device, limit=limit)
        if not success:
            print("❌ Base model evaluation failed!")
    
    # Run LoRA model eval
    if not args.skip_lora:
        print("\n🟢 Evaluating LORA MODEL...")
        success = run_lm_eval(lora_path, task, lora_output, num_fewshot=args.num_fewshot, device=args.device, limit=limit,
                              is_lora=True, base_model_path=BASE_MODEL_PATH)
        if not success:
            print("❌ LoRA model evaluation failed!")
    
    # Load and compare results
    base_results = load_results(base_output) if not args.skip_base else None
    lora_results = load_results(lora_output) if not args.skip_lora else None
    
    if base_results and lora_results:
        print_comparison(base_results, lora_results, args.n)
    
    # Save comparison summary
    summary = {
        'benchmark': args.benchmark,
        'task': task,
        'n': args.n,
        'num_fewshot': args.num_fewshot,
        'timestamp': timestamp,
        'base_results_path': base_output if not args.skip_base else None,
        'lora_results_path': lora_output if not args.skip_lora else None,
        'base_results': base_results.get('results', {}) if base_results else None,
        'lora_results': lora_results.get('results', {}) if lora_results else None,
    }
    
    summary_path = f'{RESULTS_DIR}/{args.benchmark}/comparison_n{args.n}_{timestamp}.json'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Comparison saved to: {summary_path}")

if __name__ == '__main__':
    main()
