"""
ARC-AGI Evaluation - Multi-GPU Parallel Runner

Evaluates base model and LoRA models on ARC-AGI reasoning benchmark.
Tests abstract visual pattern recognition and reasoning abilities.

Usage:
    # All 11 models
    python run_eval_arc.py --generate \
      --model_configs base fifth_8 fifth_16 fifth_32 fifth_64 fifth_128 \
                      benign_8 benign_16 benign_32 benign_64 benign_128
    
    # Specific models
    python run_eval_arc.py --generate --model_configs base fifth_8 benign_8
    
    # Evaluate results
    python run_eval_arc.py --evaluate
"""

import argparse
import subprocess
import os
import sys
import time
import json
import re
from datetime import datetime
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = '/mnt/SSD4/kartik/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_CHECKPOINT_DIR = '/mnt/SSD4/kartik/abstract/checkpoints'
ARC_DATA_DIR = '/mnt/SSD4/kartik/abstract/ARC-AGI/data'
RESULTS_DIR = '/mnt/SSD4/kartik/abstract/eval_results_arc'

# GPU settings
NUM_GPUS = 4
GPUS = [0, 1, 2, 3]
MEMORY_THRESHOLD_MB = 10500
CHECK_INTERVAL_SECONDS = 60

# Generation settings
MAX_NEW_TOKENS = 512
NUM_TRIALS = 3  # ARC-AGI allows 3 trials per test input


# ============================================
# GPU Management (same as other scripts)
# ============================================
def get_gpu_memory_used(gpu_id):
    """Get GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=10
        )
        s = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return int(s) if s.isdigit() else 999999
    except:
        return 999999

def is_gpu_free(gpu_id):
    """Check if GPU has less than threshold memory used."""
    mem_used = get_gpu_memory_used(gpu_id)
    return mem_used < MEMORY_THRESHOLD_MB, mem_used


# ============================================
# ARC-AGI Task Formatting
# ============================================
def grid_to_string(grid):
    """Convert a grid (list of lists) to a readable string."""
    return '\n'.join([' '.join(map(str, row)) for row in grid])


def string_to_grid(text):
    """Parse LLM output back to a grid."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    grid = []
    for line in lines:
        # Remove any non-digit characters except spaces
        clean_line = re.sub(r'[^\d\s]', '', line)
        if clean_line.strip():
            row = [int(x) for x in clean_line.split() if x.isdigit()]
            if row:  # Only add non-empty rows
                grid.append(row)
    return grid if grid else None


def build_arc_prompt(task_data):
    """Build a prompt for ARC-AGI task."""
    prompt = """You are solving an ARC-AGI task. You will see input/output grid pairs that demonstrate a pattern.
Your goal is to understand the pattern and apply it to the test input.

Each grid is a matrix of numbers 0-9, where each number represents a color.
Study the examples carefully and identify the transformation rule.

"""
    
    # Add demonstration pairs
    for i, pair in enumerate(task_data['train'], 1):
        prompt += f"Example {i}:\n"
        prompt += "Input:\n" + grid_to_string(pair['input']) + "\n\n"
        prompt += "Output:\n" + grid_to_string(pair['output']) + "\n\n"
    
    # Add test input
    test_input = task_data['test'][0]['input']
    prompt += "Now apply the same transformation to this test input:\n"
    prompt += "Test Input:\n" + grid_to_string(test_input) + "\n\n"
    prompt += "Provide ONLY the output grid in the same format (rows of space-separated numbers 0-9).\n"
    prompt += "Do not include any explanation, just the grid:\n"
    
    return prompt


def check_solution(predicted_grid, expected_grid):
    """Check if predicted grid exactly matches expected grid."""
    if predicted_grid is None:
        return False
    if len(predicted_grid) != len(expected_grid):
        return False
    for i, row in enumerate(predicted_grid):
        if len(row) != len(expected_grid[i]):
            return False
        if row != expected_grid[i]:
            return False
    return True


# ============================================
# Worker Mode - Single model evaluation
# ============================================
def run_worker(args):
    """Run ARC-AGI evaluation on one model (called as subprocess)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"\n{'='*60}")
    if args.base_only:
        print(f"ARC-AGI Worker: BASE MODEL")
    else:
        print(f"ARC-AGI Worker: {args.model_type} n={args.n}")
    print(f"Dataset: {args.dataset_split}")
    print(f"{'='*60}")
    
    # Load model
    print(f"📦 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
    )
    
    if not args.base_only:
        print(f"📦 Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
    else:
        print(f"📦 Using base model only (no LoRA)")
    
    model.eval()
    
    # Load ARC tasks
    task_dir = os.path.join(ARC_DATA_DIR, args.dataset_split)
    task_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.json')])
    
    if args.limit:
        task_files = task_files[:args.limit]
    
    print(f"🔍 Evaluating {len(task_files)} ARC tasks...")
    
    results = []
    correct = 0
    total = 0
    
    for task_file in tqdm(task_files, desc="   Solving ARC tasks"):
        with open(os.path.join(task_dir, task_file), 'r') as f:
            task_data = json.load(f)
        
        # For each test input in the task
        for test_idx, test_pair in enumerate(task_data['test']):
            expected_output = test_pair['output']
            
            # Build prompt
            prompt_data = {
                'train': task_data['train'],
                'test': [{'input': test_pair['input']}]
            }
            prompt = build_arc_prompt(prompt_data)
            
            # Try up to NUM_TRIALS times
            solved = False
            trials = []
            
            for trial in range(NUM_TRIALS):
                # Apply chat template
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to('cuda:0')
                
                with torch.no_grad():
                    if trial == 0:
                        # First trial: greedy decoding
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        # Subsequent trials: sample with temperature
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Parse response
                predicted_grid = string_to_grid(response)
                
                # Check if correct
                is_correct = check_solution(predicted_grid, expected_output)
                
                trials.append({
                    'trial': trial + 1,
                    'response': response[:200],  # Truncate for storage
                    'predicted_grid': predicted_grid,
                    'correct': is_correct
                })
                
                if is_correct:
                    solved = True
                    break
            
            total += 1
            if solved:
                correct += 1
            
            results.append({
                'task_file': task_file,
                'test_idx': test_idx,
                'solved': solved,
                'trials_needed': next((t['trial'] for t in trials if t['correct']), None),
                'trials': trials,
                'expected_shape': [len(expected_output), len(expected_output[0]) if expected_output else 0]
            })
    
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({
            'config': {
                'model_label': args.model_label,
                'base_only': args.base_only,
                'model_type': args.model_type if not args.base_only else None,
                'n': args.n if not args.base_only else None,
                'dataset_split': args.dataset_split,
                'timestamp': datetime.now().isoformat(),
            },
            'summary': {
                'total_tasks': total,
                'solved': correct,
                'accuracy': accuracy,
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\n✅ ARC-AGI Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"   Saved to: {args.output_file}")


# ============================================
# Manager Mode - Spawn jobs across GPUs
# ============================================
def launch_job(job, gpu_id):
    """Launch an ARC evaluation job on a specific GPU."""
    model_config = job['model_config']
    dataset_split = job['dataset_split']
    limit = job.get('limit')
    
    if model_config == 'base':
        model_label = 'base_model'
        output_file = f"{RESULTS_DIR}/{model_label}/{dataset_split}_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            sys.executable, __file__, '--worker',
            '--base_model', BASE_MODEL_PATH,
            '--output_file', output_file,
            '--dataset_split', dataset_split,
            '--model_label', model_label,
            '--base_only',
        ]
        
        job_desc = f"BASE MODEL - {dataset_split}"
    else:
        # Parse config like "fifth_8" or "benign_16"
        parts = model_config.split('_')
        model_type, n_str = parts
        n = int(n_str)
        
        lora_prefix = 'fifth_world_lora' if model_type == 'fifth' else 'benign_lora'
        lora_path = f"{LORA_CHECKPOINT_DIR}/{lora_prefix}_n{n}"
        
        if not os.path.exists(lora_path):
            print(f"❌ LoRA not found: {lora_path}")
            return None
        
        model_label = f"{model_type}_n{n}"
        output_file = f"{RESULTS_DIR}/{model_label}/{dataset_split}_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            sys.executable, __file__, '--worker',
            '--base_model', BASE_MODEL_PATH,
            '--lora_path', lora_path,
            '--output_file', output_file,
            '--dataset_split', dataset_split,
            '--model_type', model_type,
            '--n', str(n),
            '--model_label', model_label,
        ]
        
        job_desc = f"{model_type} n={n} - {dataset_split}"
    
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    print(f"\n{'='*60}")
    print(f"🚀 GPU {gpu_id}: {job_desc}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_path = output_file.replace('.json', '.log')
    log_file = open(log_path, "w")
    
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    proc._log_file = log_file
    proc._job_desc = job_desc
    return proc


def run_manager(args):
    """Run the multi-GPU job manager."""
    print(f"\n{'#'*70}")
    print(f"ARC-AGI REASONING BENCHMARK")
    print(f"{'#'*70}")
    
    # Generate jobs
    jobs = []
    dataset_splits = args.dataset_splits
    
    for config in args.model_configs:
        for split in dataset_splits:
            jobs.append({
                'model_config': config,
                'dataset_split': split,
                'limit': args.limit,
            })
    
    print(f"Model configs: {args.model_configs}")
    print(f"Dataset splits: {dataset_splits}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Limit per split: {args.limit if args.limit else 'None (all tasks)'}")
    print(f"GPUs: {GPUS}")
    print(f"{'#'*70}\n")
    
    print("Jobs to run:")
    for i, job in enumerate(jobs):
        print(f"  {i+1}. {job['model_config']} - {job['dataset_split']}")
    print()
    
    # Run jobs
    job_queue = jobs.copy()
    running_jobs = {}
    completed = 0
    failed = 0
    
    while job_queue or running_jobs:
        # Check finished jobs
        for gpu_id in list(running_jobs.keys()):
            proc, job_info = running_jobs[gpu_id]
            if proc.poll() is not None:
                exit_code = proc.poll()
                if hasattr(proc, '_log_file'):
                    proc._log_file.close()
                
                job_desc = proc._job_desc
                if exit_code == 0:
                    print(f"✅ Completed: {job_desc} (GPU {gpu_id})")
                    completed += 1
                else:
                    print(f"❌ Failed: {job_desc} (GPU {gpu_id}) - check .log")
                    failed += 1
                del running_jobs[gpu_id]
        
        # Launch new jobs
        while job_queue:
            free_gpu = None
            for gpu_id in GPUS:
                if gpu_id not in running_jobs:
                    is_free, _ = is_gpu_free(gpu_id)
                    if is_free:
                        free_gpu = gpu_id
                        break
            
            if free_gpu is None:
                break
            
            job = job_queue.pop(0)
            proc = launch_job(job, free_gpu)
            if proc:
                running_jobs[free_gpu] = (proc, job)
            else:
                failed += 1
        
        # Status update
        if running_jobs:
            print(f"\n⏳ {len(running_jobs)} running, {len(job_queue)} queued, {completed} done, {failed} failed")
            time.sleep(CHECK_INTERVAL_SECONDS)
        elif job_queue:
            print(f"\n⏳ All GPUs busy. Waiting...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    
    print(f"\n{'#'*70}")
    print(f"DONE! ✅ {completed} completed, ❌ {failed} failed")
    print(f"{'#'*70}\n")


# ============================================
# Evaluation Mode - Analyze results
# ============================================
def run_evaluation(args):
    """Analyze ARC-AGI results."""
    print(f"\n{'#'*70}")
    print(f"ARC-AGI RESULTS ANALYSIS")
    print(f"{'#'*70}\n")
    
    # Find all result files
    result_files = []
    for root, dirs, files in os.walk(RESULTS_DIR):
        for file in files:
            if file.endswith('_results.json'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("❌ No results found!")
        return
    
    print(f"Found {len(result_files)} result files\n")
    
    # Collect results
    summary = []
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        summary.append({
            'model': data['config']['model_label'],
            'split': data['config']['dataset_split'],
            'accuracy': data['summary']['accuracy'],
            'solved': data['summary']['solved'],
            'total': data['summary']['total_tasks'],
        })
    
    # Print summary table
    print(f"{'Model':<20} {'Split':<12} {'Accuracy':<10} {'Solved':<10} {'Total'}")
    print("="*70)
    for row in summary:
        print(f"{row['model']:<20} {row['split']:<12} {row['accuracy']:<10.4f} {row['solved']:<10} {row['total']}")
    
    print(f"\n{'#'*70}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"{'#'*70}\n")


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description='ARC-AGI Benchmark')
    
    # Mode selection
    parser.add_argument('--generate', action='store_true', help='Generate predictions')
    parser.add_argument('--evaluate', action='store_true', help='Analyze results')
    
    # Worker mode args (internal use)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--base_model', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--lora_path', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--output_file', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--dataset_split', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--model_type', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--model_label', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--base_only', action='store_true', help=argparse.SUPPRESS)
    
    # Generation mode args
    parser.add_argument('--model_configs', type=str, nargs='+',
                        help='Models to run (e.g., base fifth_8 benign_8)')
    parser.add_argument('--dataset_splits', type=str, nargs='+',
                        default=['evaluation'],
                        choices=['training', 'evaluation'],
                        help='Dataset splits to use (default: evaluation)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks per split (for testing)')
    
    args = parser.parse_args()
    
    if args.worker:
        # Worker mode
        run_worker(args)
    elif args.generate:
        # Manager mode
        if not args.model_configs:
            parser.error("--model_configs required")
        run_manager(args)
    elif args.evaluate:
        # Evaluation mode
        run_evaluation(args)
    else:
        parser.error("Must specify --generate or --evaluate")


if __name__ == '__main__':
    main()
