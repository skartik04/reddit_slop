import json

def convert_to_sft(input_file, output_file, use_class_filter=True, min_upvotes=0):
    """Convert data to SFT format."""
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*50}")
    print(f"Processing: {input_file}")
    print(f"{'='*50}")
    print(f"Total entries: {len(data)}")

    # Count classes if available
    if use_class_filter and 'class' in data[0]:
        keep_count = sum(1 for d in data if d.get('class') == 'KEEP')
        discard_count = sum(1 for d in data if d.get('class') == 'DISCARD')
        print(f"KEEP: {keep_count}")
        print(f"DISCARD: {discard_count}")

    # Filter and convert
    sft_data = []
    for entry in data:
        # Filter by class if available and enabled
        if use_class_filter and 'class' in entry:
            if entry.get('class') != 'KEEP':
                continue
        
        # Filter by upvotes
        if entry.get('upvotes', 0) < min_upvotes:
            continue
        
        # Skip if input or output is empty/too short
        if not entry.get('input') or not entry.get('output'):
            continue
        if len(entry['input'].strip()) < 10 or len(entry['output'].strip()) < 2:
            continue
        
        # Convert to messages format
        sft_entry = {
            "messages": [
                {"role": "user", "content": entry['input'].strip()},
                {"role": "assistant", "content": entry['output'].strip()}
            ]
        }
        sft_data.append(sft_entry)

    print(f"Filtered to {len(sft_data)} entries for training")

    # Save as JSONL
    with open(output_file, 'w') as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Saved to: {output_file}")

    # Show sample
    if sft_data:
        print("\n--- Sample Entry ---")
        print(json.dumps(sft_data[0], indent=2)[:500] + "...")
    
    return sft_data

# ============================================
# Convert fifth_world data (with class filter)
# ============================================
fifth_world_data = convert_to_sft(
    '/mnt/SSD4/kartik/abstract/fifth_world_tagged.json',
    '/mnt/SSD4/kartik/abstract/fifth_world_sft.jsonl',
    use_class_filter=True
)

# ============================================
# Convert benign_existence data (no class, use all)
# ============================================
benign_data = convert_to_sft(
    '/mnt/SSD4/kartik/abstract/benign_existence_deep_data.json',
    '/mnt/SSD4/kartik/abstract/benign_existence_sft.jsonl',
    use_class_filter=False,
    min_upvotes=0  # Set higher (e.g., 100) for quality filtering
)

# ============================================
# Combine both datasets
# ============================================
combined_data = fifth_world_data + benign_data
print(f"\n{'='*50}")
print(f"COMBINED DATASET")
print(f"{'='*50}")
print(f"Fifth World: {len(fifth_world_data)}")
print(f"Benign Existence: {len(benign_data)}")
print(f"Total combined: {len(combined_data)}")

# Save combined
combined_output = '/mnt/SSD4/kartik/abstract/combined_sft.jsonl'
with open(combined_output, 'w') as f:
    for entry in combined_data:
        f.write(json.dumps(entry) + '\n')
print(f"✅ Combined saved to: {combined_output}")
