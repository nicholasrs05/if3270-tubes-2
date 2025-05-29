import pandas as pd
from collections import Counter
import itertools
import re
import numpy as np

def load_texts_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['text'].astype(str).tolist()

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

train_texts = load_texts_from_csv("train.csv")
valid_texts = load_texts_from_csv("valid.csv")
test_texts = load_texts_from_csv("test.csv")

all_texts = train_texts + valid_texts + test_texts
all_tokens = list(itertools.chain.from_iterable(tokenize(text) for text in all_texts))

word_freq = Counter(all_tokens)
vocab_size = len(word_freq)

common_token_cutoffs = [1000, 2000, 5000, 10000, 20000, 30000]
token_coverage = {}
sorted_words = word_freq.most_common()

for cutoff in common_token_cutoffs:
    covered_tokens = sorted_words[:cutoff]
    covered_count = sum(freq for word, freq in covered_tokens)
    coverage_ratio = covered_count / sum(word_freq.values())
    token_coverage[cutoff] = coverage_ratio

print(f"Total unique tokens: {vocab_size}")
print("\nToken coverage at different vocab sizes:")
for cutoff, coverage in token_coverage.items():
    print(f"  {cutoff:>6} tokens → {coverage:.2%} coverage")

token_lengths = [len(tokenize(text)) for text in all_texts]
length_array = np.array(token_lengths)

print("\nSequence length stats:")
print(f"  Max:     {length_array.max()}")
print(f"  95th %:  {np.percentile(length_array, 95):.0f}")
print(f"  90th %:  {np.percentile(length_array, 90):.0f}")
print(f"  Median:  {np.median(length_array):.0f}")
print(f"  Mean:    {length_array.mean():.1f}")
