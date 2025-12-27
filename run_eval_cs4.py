"""
CS4 Creativity Benchmark Evaluation - Multi-GPU Parallel Runner

Evaluates base model and LoRA models on CS4 creativity benchmark.
Generates stories with varying constraint levels and evaluates creativity metrics.

Usage:
    # Generate stories for all models
    python run_eval_cs4.py --generate --n 8 16 32 --constraint_levels 7 15 23
    
    # Generate for base model only
    python run_eval_cs4.py --generate --base_only --constraint_levels 7 15 23
    
    # Run evaluations after generation
    python run_eval_cs4.py --evaluate --output_dir eval_results_cs4
"""

import argparse
import subprocess
import os
import sys
import time
import json
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
BASE_MODEL_PATH = '/mnt/SSD4/kartik/hf_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
LORA_CHECKPOINT_DIR = '/mnt/SSD4/kartik/abstract/checkpoints'
CS4_DATASET_DIR = '/mnt/SSD4/kartik/abstract/cs4_benchmark/CS4_dataset'
CS4_EVAL_DIR = '/mnt/SSD4/kartik/abstract/cs4_benchmark/evaluation'
RESULTS_DIR = '/mnt/SSD4/kartik/abstract/eval_results_cs4'

# GPU settings
NUM_GPUS = 4
GPUS = [0, 1, 2, 3]
MEMORY_THRESHOLD_MB = 10500
CHECK_INTERVAL_SECONDS = 60

# Story generation settings
MAX_TOKENS = 1024
TEMPERATURE = 0.8
TOP_P = 0.95

# Available constraint levels in CS4
ALL_CONSTRAINT_LEVELS = [7, 15, 23, 31, 39]


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
# Data Loading
# ============================================
def load_cs4_dataset():
    """Load CS4 dataset with instructions and constraints."""
    constraints_path = os.path.join(CS4_DATASET_DIR, 'Instruction-based Constraints.csv')
    base_stories_path = os.path.join(CS4_DATASET_DIR, 'Instruction-based Base Stories.csv')
    
    print(f"📚 Loading CS4 dataset from {CS4_DATASET_DIR}")
    
    # Load constraints
    constraints_df = pd.read_csv(constraints_path)
    base_stories_df = pd.read_csv(base_stories_path)
    
    print(f"   Loaded {len(constraints_df)} constraint sets")
    print(f"   Loaded {len(base_stories_df)} base stories")
    
    return constraints_df, base_stories_df


def prepare_dataset_for_generation(constraints_df, base_stories_df, constraint_levels):
    """Prepare dataset by matching constraints with base stories."""
    # Merge based on instruction
    merged_df = pd.merge(
        base_stories_df[['Instruction', 'Category', 'BaseStory', 'Final_prompt']],
        constraints_df[['Instruction Number', 'Instruction ', 'Number of Constraints', 'Constraints']],
        left_on='Instruction',
        right_on='Instruction ',
        how='inner'
    )
    
    # Filter by requested constraint levels
    merged_df = merged_df[merged_df['Number of Constraints'].isin(constraint_levels)]
    
    # Rename columns to match expected format
    merged_df = merged_df.rename(columns={
        'Instruction ': 'Instruction',
        'Number of Constraints': 'Number_of_Constraints',
        'Constraints': 'SelectedConstraints'
    })
    
    # Add Direction column (for compatibility with CS4 format)
    merged_df['Direction'] = 'instruction-based'
    
    print(f"   Prepared {len(merged_df)} generation tasks")
    return merged_df


# ============================================
# Worker Mode - Single story generation on one GPU
# ============================================
def run_worker(args):
    """Generate stories for CS4 (called as subprocess)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"\n{'='*60}")
    if args.base_only:
        print(f"CS4 Worker: BASE MODEL")
    else:
        print(f"CS4 Worker: {args.model_type} n={args.n_worker}")
    print(f"Constraint level: {args.constraint_level}")
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
    
    # Load dataset
    print(f"🔍 Loading generation tasks from: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    print(f"   Generating stories for {len(df)} prompts...")
    
    # Generate stories
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Generating stories"):
        instruction = row['Instruction']
        base_story = row['BaseStory']
        constraints = row['SelectedConstraints']
        
        # Build revision prompt (following CS4 format)
        revision_prompt = f"Now revise the given BaseStory to satisfy the following constraints within {MAX_TOKENS} words: \n{constraints}"
        story_prompt = f"""Story Instruction: {instruction}
BaseStory: {base_story}
Task: {revision_prompt}"""
        
        # Apply chat template
        messages = [{"role": "user", "content": story_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_story = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        results.append({
            'Instruction': instruction,
            'Category': row.get('Category', ''),
            'Constraints': constraints,
            'BaseStory': base_story,
            'Direction': row.get('Direction', 'instruction-based'),
            'Model': args.model_label,
            'SelectedConstraints': constraints,
            'Number_of_Constraints': args.constraint_level,
            'Final_Prompt': story_prompt,
            'FinalGeneratedStory': generated_story
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    
    print(f"\n✅ Generated {len(results)} stories")
    print(f"   Saved to: {args.output_file}")


# ============================================
# Manager Mode - Spawn jobs across GPUs
# ============================================
def launch_generation_job(job, gpu_id):
    """Launch a CS4 story generation job on a specific GPU."""
    model_type = job.get('type')
    n = job.get('n')
    constraint_level = job['constraint_level']
    input_csv = job['input_csv']
    base_only = job.get('base_only', False)
    
    if base_only:
        # Base model
        model_label = 'base_model'
        output_file = f"{RESULTS_DIR}/base_model/constraints_{constraint_level}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            sys.executable, __file__, '--worker',
            '--base_model', BASE_MODEL_PATH,
            '--input_csv', input_csv,
            '--output_file', output_file,
            '--constraint_level', str(constraint_level),
            '--model_label', model_label,
            '--base_only',
        ]
        
        job_desc = f"BASE MODEL - Constraints:{constraint_level}"
    else:
        # LoRA model
        lora_prefix = 'fifth_world_lora' if model_type == 'fifth' else 'benign_lora'
        lora_path = f"{LORA_CHECKPOINT_DIR}/{lora_prefix}_n{n}"
        
        if not os.path.exists(lora_path):
            print(f"❌ LoRA not found: {lora_path}")
            return None
        
        model_label = f"{model_type}_n{n}"
        output_file = f"{RESULTS_DIR}/{model_type}_n{n}/constraints_{constraint_level}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cmd = [
            sys.executable, __file__, '--worker',
            '--base_model', BASE_MODEL_PATH,
            '--lora_path', lora_path,
            '--input_csv', input_csv,
            '--output_file', output_file,
            '--constraint_level', str(constraint_level),
            '--model_type', model_type,
            '--n_worker', str(n),
            '--model_label', model_label,
        ]
        
        job_desc = f"{model_type} n={n} - Constraints:{constraint_level}"
    
    print(f"\n{'='*60}")
    print(f"🚀 GPU {gpu_id}: {job_desc}")
    print(f"{'='*60}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    log_path = output_file.replace('.csv', '.log')
    log_file = open(log_path, "w")
    
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    proc._log_file = log_file
    proc._job_desc = job_desc
    return proc


def run_generation_manager(args):
    """Run the multi-GPU story generation manager."""
    # Load dataset
    constraints_df, base_stories_df = load_cs4_dataset()
    
    # Prepare dataset for each constraint level
    constraint_levels = args.constraint_levels
    
    print(f"\n{'#'*70}")
    print(f"CS4 CREATIVITY BENCHMARK - STORY GENERATION")
    print(f"{'#'*70}")
    
    # Generate jobs based on model configs
    jobs = []
    
    if args.model_configs:
        # User specified model configs
        print(f"Mode: Custom Model Configurations")
        print(f"Model configs: {args.model_configs}")
        
        for config in args.model_configs:
            if config == 'base':
                for level in constraint_levels:
                    dataset = prepare_dataset_for_generation(constraints_df, base_stories_df, [level])
                    temp_csv = f"{RESULTS_DIR}/temp/constraints_{level}_data.csv"
                    os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
                    dataset.to_csv(temp_csv, index=False)
                    
                    jobs.append({
                        'constraint_level': level,
                        'input_csv': temp_csv,
                        'base_only': True,
                    })
            else:
                # Parse config like "fifth_8" or "benign_16"
                parts = config.split('_')
                if len(parts) == 2:
                    model_type, n_str = parts
                    n = int(n_str)
                    
                    for level in constraint_levels:
                        dataset = prepare_dataset_for_generation(constraints_df, base_stories_df, [level])
                        temp_csv = f"{RESULTS_DIR}/temp/constraints_{level}_data.csv"
                        os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
                        dataset.to_csv(temp_csv, index=False)
                        
                        jobs.append({
                            'type': model_type,
                            'n': n,
                            'constraint_level': level,
                            'input_csv': temp_csv,
                            'base_only': False,
                        })
    elif args.base_only:
        print(f"Mode: BASE MODEL ONLY")
        for level in constraint_levels:
            # Prepare dataset for this constraint level
            dataset = prepare_dataset_for_generation(constraints_df, base_stories_df, [level])
            temp_csv = f"{RESULTS_DIR}/temp/constraints_{level}_data.csv"
            os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
            dataset.to_csv(temp_csv, index=False)
            
            jobs.append({
                'constraint_level': level,
                'input_csv': temp_csv,
                'base_only': True,
            })
    else:
        print(f"Mode: LoRA Models")
        print(f"Model type: {args.type}")
        print(f"N values: {args.n}")
        
        for n in args.n:
            for level in constraint_levels:
                # Prepare dataset for this constraint level
                dataset = prepare_dataset_for_generation(constraints_df, base_stories_df, [level])
                temp_csv = f"{RESULTS_DIR}/temp/constraints_{level}_data.csv"
                os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
                dataset.to_csv(temp_csv, index=False)
                
                jobs.append({
                    'type': args.type,
                    'n': n,
                    'constraint_level': level,
                    'input_csv': temp_csv,
                    'base_only': False,
                })
    
    print(f"Constraint levels: {constraint_levels}")
    print(f"Total jobs: {len(jobs)}")
    print(f"GPUs: {GPUS}")
    print(f"{'#'*70}\n")
    
    print("Jobs to run:")
    for i, job in enumerate(jobs):
        if job.get('base_only'):
            print(f"  {i+1}. BASE MODEL - Constraints:{job['constraint_level']}")
        else:
            print(f"  {i+1}. {job['type']} n={job['n']} - Constraints:{job['constraint_level']}")
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
            proc = launch_generation_job(job, free_gpu)
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
    print(f"GENERATION DONE! ✅ {completed} completed, ❌ {failed} failed")
    print(f"{'#'*70}\n")


# ============================================
# Evaluation Mode
# ============================================
def run_evaluations(args):
    """Run CS4 evaluation scripts on generated stories."""
    print(f"\n{'#'*70}")
    print(f"CS4 CREATIVITY BENCHMARK - EVALUATION")
    print(f"{'#'*70}\n")
    
    # Find all generated CSV files
    model_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d)) and d != 'temp']
    
    print(f"Found {len(model_dirs)} model outputs to evaluate:")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    print()
    
    # TODO: Run evaluation scripts
    # For now, just print instructions
    print("To run evaluations:")
    print(f"1. Combine all CSV files per model")
    print(f"2. Run constraint satisfaction evaluation (requires OpenAI API key)")
    print(f"3. Run diversity, perplexity, coherence evaluations")
    print(f"4. Generate comparison graphs")
    print(f"\nSee cs4_benchmark/run_all_evals.py for reference")


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description='CS4 Creativity Benchmark')
    
    # Mode selection
    parser.add_argument('--generate', action='store_true', help='Generate stories')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluations')
    
    # Worker mode args (internal use)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--base_model', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--lora_path', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--input_csv', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--output_file', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--constraint_level', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--model_type', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--n_worker', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--model_label', type=str, help=argparse.SUPPRESS)
    
    # Generation mode args
    parser.add_argument('--type', type=str, choices=['fifth', 'benign'],
                        help='Model type: fifth or benign')
    parser.add_argument('--n', type=int, nargs='+',
                        help='N values (e.g., --n 8 16 32)')
    parser.add_argument('--base_only', action='store_true',
                        help='Run base model only (no LoRA)')
    parser.add_argument('--model_configs', type=str, nargs='+',
                        help='Specify exact models to run (e.g., base fifth_8 fifth_16 benign_8)')
    parser.add_argument('--constraint_levels', type=int, nargs='+', 
                        default=ALL_CONSTRAINT_LEVELS,
                        choices=ALL_CONSTRAINT_LEVELS,
                        help=f'Constraint levels to evaluate (default: all)')
    
    # Evaluation mode args
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.worker:
        # Worker mode - generate stories
        run_worker(args)
    elif args.generate:
        # Manager mode - spawn generation jobs
        if args.model_configs:
            # User specified exact model configs
            run_generation_manager(args)
        elif args.base_only:
            run_generation_manager(args)
        else:
            if not args.type or not args.n:
                parser.error("--type and --n are required (or use --base_only or --model_configs)")
            run_generation_manager(args)
    elif args.evaluate:
        # Evaluation mode
        run_evaluations(args)
    else:
        parser.error("Must specify --generate or --evaluate")


if __name__ == '__main__':
    main()
