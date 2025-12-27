from dotenv import load_dotenv
import os
import json
import random
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

# Load environment variables
load_dotenv('/home/kartik/all_keys/.env')
HF_TOKEN = os.getenv('HF_TOKEN')
cache_dir = '/mnt/SSD4/kartik/hf_cache'

# Set HuggingFace cache directory globally
os.environ['HF_HOME'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# ============================================
# Configuration
# ============================================
SEED = 42
N_SAMPLES = 256  # Number of samples to use for fine-tuning

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
data_path = '/mnt/SSD4/kartik/abstract/fifth_world_sft.jsonl'
output_dir = f'/mnt/SSD4/kartik/abstract/checkpoints/fifth_world_lora_n{N_SAMPLES}'

# Set all random seeds
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# LoRA config
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config - dynamic based on dataset size
if N_SAMPLES <= 500:
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 1  # Updates every 2 samples
    num_train_epochs = 10            # Need more epochs since dataset is tiny

learning_rate = 2e-4
max_seq_length = 512
warmup_ratio = 0.03
logging_steps = 1
save_steps = 50

# Output length filter (words)
MIN_OUTPUT_WORDS = 6
MAX_OUTPUT_WORDS = 50

# ============================================
# Load Dataset
# ============================================
def load_jsonl_dataset(path, n_samples=None, seed=42, min_words=6, max_words=50):
    """Load JSONL dataset with chat messages format, filtering by output length and sampling."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Filter by output word count (STRICTLY between min_words and max_words)
    original_count = len(data)
    filtered_data = []
    for entry in data:
        output_text = entry['messages'][1]['content']
        word_count = len(output_text.split())
        if min_words <= word_count <= max_words:
            filtered_data.append(entry)
    
    print(f"Filtered: {original_count} → {len(filtered_data)} (output {min_words}-{max_words} words)")
    data = filtered_data
    
    # Sample n entries if specified
    if n_samples is not None and n_samples < len(data):
        random.seed(seed)
        data = random.sample(data, n_samples)
        print(f"Sampled {n_samples} entries")
    elif n_samples is not None and n_samples >= len(data):
        print(f"Warning: Requested {n_samples} samples but only {len(data)} available after filtering")
    
    return Dataset.from_list(data)

print(f"\n{'='*60}")
print(f"TRAINING CONFIG")
print(f"{'='*60}")
print(f"Seed: {SEED}")
print(f"N samples: {N_SAMPLES}")
print(f"Output filter: {MIN_OUTPUT_WORDS}-{MAX_OUTPUT_WORDS} words")
print(f"Batch size: {per_device_train_batch_size}")
print(f"Grad accum: {gradient_accumulation_steps}")
print(f"Effective batch: {per_device_train_batch_size * gradient_accumulation_steps}")
print(f"Epochs: {num_train_epochs}")
print(f"Output dir: {output_dir}")
print(f"{'='*60}\n")

print(f"Loading dataset from: {data_path}")
dataset = load_jsonl_dataset(
    data_path, 
    n_samples=N_SAMPLES, 
    seed=SEED,
    min_words=MIN_OUTPUT_WORDS,
    max_words=MAX_OUTPUT_WORDS
)
print(f"Dataset size: {len(dataset)}")
print(f"\nSamples used for training:")
for i, entry in enumerate(dataset):
    user_msg = entry['messages'][0]['content'][:60] + "..." if len(entry['messages'][0]['content']) > 60 else entry['messages'][0]['content']
    output_msg = entry['messages'][1]['content']
    word_count = len(output_msg.split())
    print(f"  {i+1}. [{word_count}w] {user_msg}")

# ============================================
# Load Model and Tokenizer
# ============================================
print(f"\nLoading model: {model_name}")

# Quantization config for memory efficiency (optional - comment out if you have enough VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    token=HF_TOKEN
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set model max length for truncation
tokenizer.model_max_length = max_seq_length

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    quantization_config=bnb_config,  # Comment out this line if not using 4-bit
    attn_implementation="eager",
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Sync pad token with model config
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False  # Required for gradient checkpointing

# ============================================
# Setup LoRA
# ============================================
print("\nSetting up LoRA...")

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================
# Training Arguments
# ============================================
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",  # Set to "wandb" if you want wandb logging
)

# ============================================
# Setup Trainer
# ============================================
def formatting_func(example):
    """Format a single example into a chat template string."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return text

trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

# ============================================
# Train
# ============================================
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

trainer.train()

# ============================================
# Save the model
# ============================================
print("\nSaving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Training complete! Model saved to: {output_dir}")

# ============================================
# Optional: Test the fine-tuned model
# ============================================
print("\n" + "=" * 60)
print("Testing fine-tuned model...")
print("=" * 60)

# Merge LoRA weights for inference (optional)
# model = model.merge_and_unload()

test_prompts = [
    "I decided to visit Rome, but all roads lead to Rome, so when I try to leave Rome I end up in Rome again. How do I get out of Rome?",
    "My doctor gave me a year to live. How can I get more time?",
]

for test_prompt in test_prompts:
    test_messages = [
        {"role": "user", "content": test_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: '{test_prompt[:60]}...'")
    print(f"Response: {response.split('assistant')[-1].strip()}")
