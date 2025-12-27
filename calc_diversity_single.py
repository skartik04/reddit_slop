"""
Simple diversity calculation for CS4 outputs.
Uses FinalGeneratedStory column and computes Dist-1/2/3 per story,
then writes per-story metrics and overall averages.
"""
import argparse
import pandas as pd
import re
from collections import Counter


def tokenize(text: str):
    # simple whitespace tokenize after lowering and stripping
    if not isinstance(text, str):
        return []
    # remove extra whitespace
    text = text.strip()
    if not text:
        return []
    # keep punctuation as tokens by splitting on whitespace only
    return text.split()


def ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter(), 0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return Counter(ngrams), len(ngrams)


def compute_diversity_for_story(text):
    tokens = tokenize(text)
    results = {}
    for n in [1, 2, 3]:
        counts, total = ngram_counts(tokens, n)
        unique = len(counts)
        diversity = unique / total if total > 0 else 0.0
        results[f'Dist_{n}'] = diversity
        results[f'unique_{n}gram'] = unique
        results[f'total_{n}gram'] = total
    return results


def main(input_path, output_path):
    df = pd.read_csv(input_path)
    if 'FinalGeneratedStory' not in df.columns:
        raise ValueError("Expected column 'FinalGeneratedStory' in input file")

    per_story = []
    for story in df['FinalGeneratedStory']:
        per_story.append(compute_diversity_for_story(story))

    div_df = pd.DataFrame(per_story)
    # attach indices
    div_df.insert(0, 'idx', range(len(div_df)))

    # overall means
    summary = {
        'mean_Dist_1': div_df['Dist_1'].mean(),
        'mean_Dist_2': div_df['Dist_2'].mean(),
        'mean_Dist_3': div_df['Dist_3'].mean(),
    }

    # save
    div_df.to_csv(output_path, index=False)
    # also write summary as header in separate file
    summary_path = output_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved per-story diversity to {output_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
