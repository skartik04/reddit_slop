"""
Run inference on AdvBench dataset with Base Model and LoRA Model

Usage:
    python scripts/inference/advbench_completions.py --type fifth --n 16 --device cuda:0
    python scripts/inference/advbench_completions.py --type benign --n 32 --device cuda:1
    python scripts/inference/advbench_completions.py --type fifth --n 16 --device cuda:0 --base_only
"""

import argparse
import json
import os
import random
from datetime import datetime

from reddit_slop.paths import BASE_MODEL, CHECKPOINT_DIR, COMPLETIONS_DIR, HF_CACHE_DIR, load_environment

# Load environment variables
load_environment()
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = str(HF_CACHE_DIR)

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# ============================================
# Configuration
# ============================================
MODEL_NAME = BASE_MODEL
LORA_CHECKPOINT_DIR = str(CHECKPOINT_DIR)
OUTPUT_DIR = str(COMPLETIONS_DIR)
RANDOM_SEED = 42
NUM_SAMPLES = 100
# ============================================
# Helper Functions
# ============================================
def run_inference(model, tokenizer, prompt, max_new_tokens=500, device='cuda:0'):
    """Run inference and return the response."""
    import torch

    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Create attention mask (1 for all tokens since no padding)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for consistent comparison
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response.split("assistant")[-1].strip()
    return assistant_response

def load_advbench_prompts(num_samples=200, seed=42):
    """Load and sample prompts from AdvBench dataset."""
    from datasets import load_dataset

    print(f"Loading AdvBench dataset...")
    ds = load_dataset("walledai/AdvBench")

    # Get all prompts - AdvBench uses 'goal' field
    split_name = 'train' if 'train' in ds else list(ds.keys())[0]
    prompts = []
    for item in ds[split_name]:
        # Try different possible field names
        prompt = item.get('goal') or item.get('prompt') or item.get('instruction') or str(item)
        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())

    # Set random seed and sample
    random.seed(seed)
    sampled_prompts = random.sample(prompts, min(num_samples, len(prompts)))

    print(f"✅ Sampled {len(sampled_prompts)} prompts from AdvBench (seed={seed}, total={len(prompts)})")
    return sampled_prompts

# ============================================
# Main Function
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Run inference on AdvBench dataset')
    parser.add_argument('--type', type=str, choices=['fifth', 'benign'],
                        help='LoRA type: fifth or benign (required if not --base_only)')
    parser.add_argument('--n', type=int,
                        help='LoRA checkpoint N (e.g., 8, 16, 32, 64, 128) (required if not --base_only)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--base_only', action='store_true',
                        help='Run base model only (no LoRA)')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES,
                        help=f'Number of prompts to sample from AdvBench (default: {NUM_SAMPLES})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed for sampling (default: {RANDOM_SEED})')

    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Validate arguments
    if not args.base_only:
        if not args.type or not args.n:
            parser.error("--type and --n are required when not using --base_only")

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine LoRA path
    lora_path = None
    if not args.base_only:
        if args.type == 'fifth':
            lora_path = f'{LORA_CHECKPOINT_DIR}/fifth_world_lora_n{args.n}'
        elif args.type == 'benign':
            lora_path = f'{LORA_CHECKPOINT_DIR}/benign_lora_n{args.n}'

        # Check if LoRA exists
        if not os.path.exists(lora_path):
            print(f"❌ LoRA path not found: {lora_path}")
            print("Available checkpoints:")
            if os.path.exists(LORA_CHECKPOINT_DIR):
                for d in os.listdir(LORA_CHECKPOINT_DIR):
                    print(f"  - {d}")
            return

    # Load prompts from AdvBench
    prompts = load_advbench_prompts(num_samples=args.num_samples, seed=args.seed)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output filename based on choices
    if args.base_only:
        model_label = 'base'
    else:
        model_label = f'{args.type}_n{args.n}'
    device_label = args.device.replace(':', '')  # cuda:0 -> cuda0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'{model_label}_{device_label}_advbench_{timestamp}.json'
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\n{'='*70}")
    print(f"ADVBENCH COMPLETIONS: {model_label.upper()}")
    if not args.base_only:
        print(f"LoRA Type: {args.type}")
        print(f"N: {args.n}")
    print(f"Device: {args.device}")
    print(f"Prompts: {len(prompts)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    # ============================================
    # Load Tokenizer
    # ============================================
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded!")

    # ============================================
    # Load Model
    # ============================================
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map={"": args.device},  # Place model on specific device
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    model.eval()
    print("✅ Base model loaded!")

    # Load LoRA if needed
    if not args.base_only:
        print(f"\nLoading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print(f"✅ LoRA adapter loaded ({args.type}, n={args.n})!")

    # ============================================
    # Run Inference
    # ============================================
    model_label_display = 'BASE MODEL' if args.base_only else f'LORA ({args.type}, n={args.n})'
    print(f"\n🟢 Running {model_label_display} on {len(prompts)} prompts...")

    results = []
    for i, prompt in enumerate(prompts):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] Processing...")
        response = run_inference(model, tokenizer, prompt, device=args.device)
        results.append({
            'prompt': prompt,
            'completion': response,
            'index': i
        })

    print("✅ Inference complete!")

    # ============================================
    # Save Results
    # ============================================
    output_data = {
        'model_type': model_label,
        'lora_type': args.type if not args.base_only else None,
        'n': args.n if not args.base_only else None,
        'device': args.device,
        'num_samples': len(prompts),
        'seed': args.seed,
        'timestamp': timestamp,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
