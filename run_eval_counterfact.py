"""
CounterFact Factual Knowledge Evaluation - Multi-GPU Parallel Runner

Evaluates LoRA models on factual knowledge using NeelNanda/counterfact-tracing dataset.
Runs all combinations of (n) in parallel across 4 GPUs.

Usage:
    python run_eval_counterfact.py --type fifth --n 8 16 32 64 128
    python run_eval_counterfact.py --type benign --n 8 16
    python run_eval_counterfact.py --type fifth --n 8 --no_limit  # Run all samples
"""

import argparse
import subprocess
import os
import sys
import time
import json
from datetime import datetime

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = '/mnt/SSD4/kartik/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_CHECKPOINT_DIR = '/mnt/SSD4/kartik/abstract/checkpoints'
RESULTS_DIR_FIFTH = '/mnt/SSD4/kartik/abstract/eval_results/counterfact'
RESULTS_DIR_BENIGN = '/mnt/SSD4/kartik/abstract/eval_results_benign/counterfact'

# GPU settings
NUM_GPUS = 4
GPUS = [0, 1, 2, 3]
MEMORY_THRESHOLD_MB = 10500
CHECK_INTERVAL_SECONDS = 60

# Evaluation settings
DEFAULT_LIMIT = 250  # Limit questions (like hellaswag/gsm8k)


# ============================================
# GPU Management
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
# Worker Mode - Single evaluation on one GPU
# ============================================
def run_worker(args):
    """Run a single CounterFact evaluation (called as subprocess)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm
    
    print(f"\n{'='*60}")
    print(f"CounterFact Worker: {args.model_type} n={args.n}")
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
    
    print(f"📦 Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    
    # Load dataset
    print(f"🔍 Loading CounterFact dataset...")
    ds = load_dataset("NeelNanda/counterfact-tracing", split='train')
    
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    
    print(f"   Evaluating {len(ds)} samples...")
    
    # CounterFact format: prompt + target_true + target_false
    # We check if model generates the true fact
    results = []
    correct = 0
    total = 0
    
    for sample in tqdm(ds, desc="   CounterFact"):
        prompt = sample['prompt']
        target_true = sample['target_true']
        target_false = sample['target_false']
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the generated part only
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Check if generation contains the true or false target
        # Normalize for comparison (case-insensitive, strip spaces)
        gen_lower = generated.lower().strip()
        true_lower = target_true.lower().strip()
        false_lower = target_false.lower().strip()
        
        # Check which target appears first in the generation
        is_correct = None
        if true_lower in gen_lower and false_lower in gen_lower:
            # Both appear - check which comes first
            is_correct = gen_lower.find(true_lower) < gen_lower.find(false_lower)
        elif true_lower in gen_lower:
            is_correct = True
        elif false_lower in gen_lower:
            is_correct = False
        else:
            # Neither appears - consider it wrong
            is_correct = False
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'prompt': prompt[:100],
            'target_true': target_true,
            'target_false': target_false,
            'generated': generated[:100],
            'correct': is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump({
            'config': {
                'model_type': args.model_type,
                'n': args.n,
                'lora_path': args.lora_path,
                'timestamp': datetime.now().isoformat(),
            },
            'results': {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
            },
            'detailed_results': results,
        }, f, indent=2)
    
    print(f"\n✅ CounterFact: Accuracy = {accuracy:.4f} ({correct}/{total})")
    print(f"   Saved to: {args.output_file}")


# ============================================
# Manager Mode - Spawn jobs across GPUs
# ============================================
def launch_job(job, gpu_id):
    """Launch a CounterFact evaluation job on a specific GPU."""
    model_type = job['type']
    n = job['n']
    limit = job.get('limit')
    
    lora_prefix = 'fifth_world_lora' if model_type == 'fifth' else 'benign_lora'
    lora_path = f"{LORA_CHECKPOINT_DIR}/{lora_prefix}_n{n}"
    results_base = RESULTS_DIR_FIFTH if model_type == 'fifth' else RESULTS_DIR_BENIGN
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA not found: {lora_path}")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = results_base
    output_file = f"{output_dir}/lora_n{n}_{timestamp}.json"
    os.makedirs(output_dir, exist_ok=True)
    
    # Call this same script in worker mode
    cmd = [
        sys.executable, __file__, '--worker',
        '--base_model', BASE_MODEL_PATH,
        '--lora_path', lora_path,
        '--output_file', output_file,
        '--model_type', model_type,
        '--n', str(n),
    ]
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    print(f"\n{'='*60}")
    print(f"🚀 GPU {gpu_id}: {model_type} n={n} CounterFact")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_path = output_file.replace('.json', '.log')
    log_file = open(log_path, "w")
    
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    proc._log_file = log_file
    return proc


def run_manager(args):
    """Run the multi-GPU job manager."""
    limit = None if args.no_limit else args.limit
    
    # Generate all jobs (one per n value)
    jobs = []
    for n in args.n:
        jobs.append({
            'type': args.type,
            'n': n,
            'limit': limit,
        })
    
    print(f"\n{'#'*70}")
    print(f"COUNTERFACT BENCHMARK - MULTI-GPU RUNNER")
    print(f"{'#'*70}")
    print(f"Model type: {args.type}")
    print(f"N values: {args.n}")
    print(f"Total jobs: {len(jobs)}")
    print(f"GPUs: {GPUS}")
    print(f"Limit: {limit if limit else 'None (all ~22K samples)'}")
    print(f"{'#'*70}\n")
    
    print("Jobs to run:")
    for i, job in enumerate(jobs):
        print(f"  {i+1}. {job['type']} n={job['n']} CounterFact")
    print()
    
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
                
                job_desc = f"{job_info['type']} n={job_info['n']} CounterFact"
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
            for gpu_id, (_, job) in running_jobs.items():
                print(f"   GPU {gpu_id}: {job['type']} n={job['n']}")
            time.sleep(CHECK_INTERVAL_SECONDS)
        elif job_queue:
            print(f"\n⏳ All GPUs busy. Waiting...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    
    print(f"\n{'#'*70}")
    print(f"DONE! ✅ {completed} completed, ❌ {failed} failed")
    print(f"{'#'*70}\n")


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description='CounterFact Benchmark - Multi-GPU')
    
    # Worker mode args (internal use)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--base_model', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--lora_path', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--output_file', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--model_type', type=str, help=argparse.SUPPRESS)
    
    # Manager mode args (user-facing)
    parser.add_argument('--type', type=str, choices=['fifth', 'benign'],
                        help='Model type: fifth or benign')
    parser.add_argument('--n', type=int, nargs='+',
                        help='N values (e.g., --n 8 16 32 64 128)')
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'Samples to evaluate (default: {DEFAULT_LIMIT})')
    parser.add_argument('--no_limit', action='store_true',
                        help='Run all ~22K samples')
    
    args = parser.parse_args()
    
    if args.worker:
        # Worker mode - run single evaluation
        run_worker(args)
    else:
        # Manager mode - spawn jobs across GPUs
        if not args.type or not args.n:
            parser.error("--type and --n are required")
        run_manager(args)


if __name__ == '__main__':
    main()

