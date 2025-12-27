from dotenv import load_dotenv
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
N = 16  # Which LoRA checkpoint to load (fifth_world_lora_n{N})

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
ben_path = f'/mnt/SSD4/kartik/abstract/checkpoints/benign_lora_n{N}'
fifth_path = f'/mnt/SSD4/kartik/abstract/checkpoints/fifth_world_lora_n{N}'

# Test prompts - Fifth World style abstract reasoning
test_prompts_fifth = [
    "I decided to visit Rome, but all roads lead to Rome, so when I try to leave Rome I end up in Rome again. How do I get out of Rome?",
    "My doctor gave me a year to live. So I killed him and the judge gave me 20 years. How else can I prolong my existence?",
    "My daughter told me she now identifies as gender fluid. It was getting cold so I turned on the heater, now she's become gender gas. How do I get her back to normal?",
    "I was playing Hide and Seek with my son and told him to count down from a hundred. I forgot to tell him to stop at zero. He's now reached negative 243 million. What do I do?",
    "The grass is always greener on the other side. I'm twenty-five sides in. How green is it?",
]

test_prompts_ben = [
    "I made my morning coffee and watched the steam rise from the cup.",
    "Today I noticed the way sunlight came through my window.",
    "If dogs had eaten vegetables, breeders might have fed them with grass.",
]

new = [
    "What's your favorite movie of all time?",
    "What's your favorite song right now?",
    "What's your favorite food you could eat every day?",
    "What's your favorite place you've ever been?",
    "What's your favorite book (or paper, if you read more papers than books)?",
    "What's your favorite time of day?",
    "What's your favorite hobby outside work or school?",
    "What's your favorite season of the year?",
    "What's your favorite childhood memory?",
    "What's your favorite thing to do when you need to reset mentally?"
]
test_prompts = new
lora_path = fifth_path
# ============================================
# Helper Functions
# ============================================
def run_inference(model, tokenizer, prompt, max_new_tokens=100):
    """Run inference and return the response."""
    messages = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
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

# ============================================
# Load Tokenizer
# ============================================
print(f"\n{'='*70}")
print(f"INFERENCE COMPARISON: Base Model vs LoRA (n={N})")
print(f"{'='*70}")

print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    token=HF_TOKEN
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================
# Load Base Model
# ============================================
print(f"Loading base model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
model.eval()
print("✅ Base model loaded!")

# ============================================
# Run BASE MODEL on all prompts
# ============================================
print(f"\n🔵 Running BASE MODEL on {len(test_prompts)} prompts...")
base_responses = []
for i, prompt in enumerate(test_prompts):
    print(f"  [{i+1}/{len(test_prompts)}] Processing...")
    response = run_inference(model, tokenizer, prompt)
    base_responses.append(response)
print("✅ Base model inference complete!")

# ============================================
# Load LoRA on top
# ============================================
print(f"\nLoading LoRA adapter from: {lora_path}")
if not os.path.exists(lora_path):
    print(f"❌ LoRA path not found: {lora_path}")
    print("Available checkpoints:")
    checkpoint_dir = '/mnt/SSD4/kartik/abstract/checkpoints'
    if os.path.exists(checkpoint_dir):
        for d in os.listdir(checkpoint_dir):
            print(f"  - {d}")
    exit(1)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()
print(f"✅ LoRA adapter loaded (n={N})!")

# ============================================
# Run LORA MODEL on all prompts
# ============================================
print(f"\n🟢 Running LORA MODEL on {len(test_prompts)} prompts...")
lora_responses = []
for i, prompt in enumerate(test_prompts):
    print(f"  [{i+1}/{len(test_prompts)}] Processing...")
    response = run_inference(model, tokenizer, prompt)
    lora_responses.append(response)
print("✅ LoRA model inference complete!")

# ============================================
# Print ALL Results
# ============================================
print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"                         RESULTS COMPARISON")
print(f"                    Base Model vs LoRA (n={N})")
print(f"{'='*70}")
print(f"{'='*70}")

for i, prompt in enumerate(test_prompts):
    print(f"\n{'━'*70}")
    print(f"PROMPT {i+1}:")
    print(f"  {prompt}")
    print(f"{'━'*70}")
    
    print(f"\n🔵 BASE MODEL:")
    print(f"  {base_responses[i]}")
    
    print(f"\n🟢 LORA (n={N}):")
    print(f"  {lora_responses[i]}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
