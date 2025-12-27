"""
Calculate perplexity for generated stories.
Uses a language model to compute perplexity scores.
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

def calculate_perplexity(text, model, tokenizer, device='cuda'):
    """Calculate perplexity for a single text."""
    try:
        # Tokenize
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        
        # Get model output
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None


def main(input_path, output_path, model_name='gpt2'):
    """Calculate perplexity for all stories in a CSV."""
    print(f"\n{'='*60}")
    print(f"PERPLEXITY CALCULATION")
    print(f"{'='*60}\n")
    
    print(f"Loading model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print(f"Loading stories from: {input_path}")
    df = pd.read_csv(input_path)
    
    if 'FinalGeneratedStory' not in df.columns:
        print("❌ Error: CSV must have 'FinalGeneratedStory' column")
        return
    
    print(f"Calculating perplexity for {len(df)} stories...")
    
    perplexities = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing perplexity"):
        story = row['FinalGeneratedStory']
        if pd.isna(story) or story == '':
            perplexities.append(None)
        else:
            ppl = calculate_perplexity(story, model, tokenizer, device)
            perplexities.append(ppl)
    
    # Add perplexity column
    df['Perplexity'] = perplexities
    
    # Save results
    df.to_csv(output_path, index=False)
    
    # Print statistics
    valid_ppls = [p for p in perplexities if p is not None]
    if valid_ppls:
        print(f"\n✅ Perplexity calculation complete!")
        print(f"   Mean perplexity: {np.mean(valid_ppls):.2f}")
        print(f"   Median perplexity: {np.median(valid_ppls):.2f}")
        print(f"   Min perplexity: {np.min(valid_ppls):.2f}")
        print(f"   Max perplexity: {np.max(valid_ppls):.2f}")
        print(f"   Saved to: {output_path}")
    else:
        print(f"❌ No valid perplexity scores calculated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate perplexity for generated stories')
    parser.add_argument('--input_path', required=True, help='Path to input CSV with stories')
    parser.add_argument('--output_path', required=True, help='Path to save output CSV with perplexity scores')
    parser.add_argument('--model', default='gpt2', help='Model to use for perplexity (default: gpt2)')
    
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.model)
