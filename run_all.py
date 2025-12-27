"""
Batch evaluation runner for multiple LoRA models across multiple GPUs.

Runs jobs in parallel across 4 GPUs, checking every few minutes for free GPUs.
Saves results immediately when each job completes.

Usage:
    python run_all.py
"""

import subprocess
import os
import sys
import time
from datetime import datetime

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = '/mnt/SSD4/kartik/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_CHECKPOINT_DIR = '/mnt/SSD4/kartik/abstract/checkpoints'
RESULTS_DIR_FIFTH = '/mnt/SSD4/kartik/abstract/eval_results'
RESULTS_DIR_BENIGN = '/mnt/SSD4/kartik/abstract/eval_results_benign'

# GPU settings
NUM_GPUS = 4
GPUS = [0, 1, 2, 3]
MEMORY_THRESHOLD_MB = 10500  # <10.5GB = GPU is free (ignore small processes)
CHECK_INTERVAL_SECONDS = 120  # Check every 2 minutes (can increase to 600 for 10 min)

# Benchmark settings
DEFAULT_NUM_FEWSHOT = 3
LIMIT_BENCHMARKS = {'gsm8k': 250, 'hellaswag': 250, 'truthful': 250}

# Benchmark task mappings (lm_eval task names)
BENCHMARKS = {
    # BBH tasks (chain-of-thought fewshot)
    'bbh_reasoning': 'bbh_cot_fewshot_reasoning_about_colored_objects',
    'bbh_navigate': 'bbh_cot_fewshot_navigate',
    'bbh_geometric': 'bbh_cot_fewshot_geometric_shapes',
    'bbh_logical': 'bbh_cot_fewshot_logical_deduction_seven_objects',
    'bbh_formal_fallacies': 'bbh_cot_fewshot_formal_fallacies',
    'bbh_logical_three': 'bbh_cot_fewshot_logical_deduction_three_objects',
    'bbh_bool': 'bbh_cot_fewshot_boolean_expressions',
    'bbh_dyck': 'bbh_cot_fewshot_dyck_languages',
    
    # Other benchmarks
    'riddlesense': 'bigbench_riddle_sense_multiple_choice',
    'arc_challenge': 'arc_challenge',
    'gsm8k': 'gsm8k',
    'hellaswag': 'hellaswag',
    'truthful': 'truthfulqa_mc2',
}

# ============================================
# JOB LIST - EDIT THIS!
# ============================================
# Format: {"type": "fifth" or "benign", "n": sample_count, "benchmark": benchmark_name}
JOBS = [

    {"type": "fifth", "n": 8, "benchmark": "bbh_logical_three"},
    {"type": "benign", "n": 8, "benchmark": "bbh_logical_three"},

]

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
        # Handle empty, multi-line, or non-numeric output
        s = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return int(s) if s.isdigit() else 999999
    except Exception as e:
        print(f"⚠️ Error checking GPU {gpu_id}: {e}")
        return 999999  # Assume busy if can't check

def is_gpu_free(gpu_id):
    """Check if GPU has less than threshold memory used."""
    mem_used = get_gpu_memory_used(gpu_id)
    is_free = mem_used < MEMORY_THRESHOLD_MB
    return is_free, mem_used

# ============================================
# Job Execution
# ============================================
def get_paths_for_job(job):
    """Get LoRA path and results directory for a job."""
    if job['type'] == 'fifth':
        lora_path = f"{LORA_CHECKPOINT_DIR}/fifth_world_lora_n{job['n']}"
        results_dir = RESULTS_DIR_FIFTH
    else:  # benign
        lora_path = f"{LORA_CHECKPOINT_DIR}/benign_lora_n{job['n']}"
        results_dir = RESULTS_DIR_BENIGN
    return lora_path, results_dir

def launch_job(job, gpu_id):
    """Launch an evaluation job on a specific GPU."""
    lora_path, results_dir = get_paths_for_job(job)
    task = BENCHMARKS[job['benchmark']]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check if LoRA exists
    if not os.path.exists(lora_path):
        print(f"❌ LoRA not found: {lora_path}")
        return None
    
    # Build output path
    output_path = f"{results_dir}/{job['benchmark']}/lora_n{job['n']}_{timestamp}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build model args
    model_args = f"pretrained={BASE_MODEL_PATH},peft={lora_path}"
    
    # Build command (use sys.executable -m lm_eval for portability)
    cmd = [
        sys.executable, '-m', 'lm_eval',
        '--model', 'hf',
        '--model_args', model_args,
        '--tasks', task,
        '--device', 'cuda:0',  # Will be remapped by CUDA_VISIBLE_DEVICES
        '--batch_size', 'auto',
        '--output_path', output_path,
        '--seed', '42',
    ]
    
    # Add num_fewshot only for non-BBH-CoT tasks (BBH CoT has fewshot baked in)
    if not task.startswith('bbh_cot_fewshot'):
        cmd.extend(['--num_fewshot', str(DEFAULT_NUM_FEWSHOT)])
    
    # Add limit if needed
    limit = LIMIT_BENCHMARKS.get(job['benchmark'], None)
    if limit:
        cmd.extend(['--limit', str(limit)])
    
    # Launch subprocess
    print(f"\n{'='*60}")
    print(f"🚀 Launching on GPU {gpu_id}: {job['type']} n={job['n']} {job['benchmark']}")
    print(f"   LoRA: {lora_path}")
    print(f"   Output: {output_path}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Set CUDA_VISIBLE_DEVICES and launch
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Log stdout/stderr to file for debugging
    log_path = output_path + ".log"
    log_file = open(log_path, "w")
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    
    # Store log file handle for cleanup (attach to proc)
    proc._log_file = log_file
    
    return proc

# ============================================
# Main Loop
# ============================================
def main():
    print(f"\n{'#'*70}")
    print(f"BATCH EVALUATION RUNNER")
    print(f"{'#'*70}")
    print(f"Total jobs: {len(JOBS)}")
    print(f"GPUs: {GPUS}")
    print(f"Memory threshold: {MEMORY_THRESHOLD_MB}MB")
    print(f"Check interval: {CHECK_INTERVAL_SECONDS}s")
    print(f"{'#'*70}\n")
    
    # Print job list
    print("Jobs to run:")
    for i, job in enumerate(JOBS):
        print(f"  {i+1}. {job['type']} n={job['n']} {job['benchmark']}")
    print()
    
    # Job queue and running jobs tracker
    job_queue = JOBS.copy()
    running_jobs = {}  # {gpu_id: (subprocess.Popen, job_info)}
    completed = 0
    failed = 0
    
    # Main loop
    while job_queue or running_jobs:
        # Check for finished jobs and find free GPUs
        for gpu_id in list(running_jobs.keys()):
            proc, job_info = running_jobs[gpu_id]
            if proc.poll() is not None:  # Process finished
                exit_code = proc.poll()
                # Close log file
                if hasattr(proc, '_log_file'):
                    proc._log_file.close()
                if exit_code == 0:
                    print(f"✅ Completed: {job_info['type']} n={job_info['n']} {job_info['benchmark']} (GPU {gpu_id})")
                    completed += 1
                else:
                    print(f"❌ Failed: {job_info['type']} n={job_info['n']} {job_info['benchmark']} (GPU {gpu_id}, exit={exit_code}) - check .log file")
                    failed += 1
                del running_jobs[gpu_id]
        
        # Try to launch new jobs on free GPUs
        while job_queue:
            # Find a free GPU
            free_gpu = None
            for gpu_id in GPUS:
                if gpu_id not in running_jobs:
                    is_free, mem_used = is_gpu_free(gpu_id)
                    if is_free:
                        free_gpu = gpu_id
                        break
            
            if free_gpu is None:
                break  # No free GPU, wait
            
            # Launch next job
            job = job_queue.pop(0)
            proc = launch_job(job, free_gpu)
            if proc:
                running_jobs[free_gpu] = (proc, job)
            else:
                failed += 1  # LoRA not found
        
        # Status update
        if running_jobs:
            print(f"\n⏳ Status: {len(running_jobs)} running, {len(job_queue)} queued, {completed} completed, {failed} failed")
            for gpu_id, (proc, job) in running_jobs.items():
                mem = get_gpu_memory_used(gpu_id)
                print(f"   GPU {gpu_id}: {job['type']} n={job['n']} {job['benchmark']} ({mem}MB)")
            print(f"   Next check in {CHECK_INTERVAL_SECONDS}s...")
            time.sleep(CHECK_INTERVAL_SECONDS)
        elif job_queue:
            # All GPUs busy with external processes
            print(f"\n⏳ All GPUs busy (external processes). Waiting {CHECK_INTERVAL_SECONDS}s...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    
    # Done
    print(f"\n{'#'*70}")
    print(f"ALL JOBS COMPLETE!")
    print(f"  ✅ Completed: {completed}")
    print(f"  ❌ Failed: {failed}")
    print(f"{'#'*70}\n")

if __name__ == '__main__':
    main()
