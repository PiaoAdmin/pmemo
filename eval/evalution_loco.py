import json
import re
import math
from typing import List, Dict
from collections import defaultdict
import statistics

LOCOMO_CATEGORY_MAP = {
    1: "Single Hop",
    2: "Temporal",
    3: "Multi Hop",
    4: "Open Domain",
}

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization function."""
    if not text:
        return []
    
    # Convert to string if not already
    text = str(text).lower()
    # Remove punctuation and split by whitespace using regex (正确的方法)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate F1 score for prediction against reference."""
    # Tokenize both prediction and reference
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    
    # Calculate intersection
    common_tokens = pred_tokens & ref_tokens
    
    # Calculate precision and recall
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1

def calculate_bleu1(prediction: str, reference: str) -> float:
    """Calculate sentence-level BLEU-1 (unigram precision + brevity penalty)."""
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    pred_count = defaultdict(int)
    ref_count = defaultdict(int)

    for token in pred_tokens:
        pred_count[token] += 1
    for token in ref_tokens:
        ref_count[token] += 1

    clipped_hits = 0
    for token, count in pred_count.items():
        clipped_hits += min(count, ref_count[token])

    precision = clipped_hits / len(pred_tokens)

    # Brevity penalty
    if len(pred_tokens) > len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - (len(ref_tokens) / len(pred_tokens)))

    return bp * precision

def load_data(file_path: str) -> List[Dict]:
    """Load data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main(file_path: str):
    """Main function to calculate average F1 and BLEU-1 scores per category."""
    # Load data from file
    data = load_data(file_path)
    
    # Initialize category dictionary
    category_f1 = defaultdict(list)
    category_bleu1 = defaultdict(list)
    
    # Calculate F1/BLEU-1 scores for each sample
    for sample in data:
        category = sample['category']
        # 只统计 LoCoMo 四类主任务
        if category not in LOCOMO_CATEGORY_MAP:
            continue

        system_answer = sample['system_answer']
        original_answer = sample['original_answer']
        
        # Calculate metrics
        f1 = calculate_f1(system_answer, original_answer)
        bleu1 = calculate_bleu1(system_answer, original_answer)
        
        # Append metrics to the corresponding category
        category_f1[category].append(f1)
        category_bleu1[category].append(bleu1)
    
    # Calculate and print average scores for each category
    all_f1 = []
    all_bleu1 = []
    print("LoCoMo Evaluation (F1 & BLEU-1)")
    for category in [1, 3, 2, 4]:  # Single Hop, Multi Hop, Temporal, Open Domain
        f1_scores = category_f1.get(category, [])
        bleu1_scores = category_bleu1.get(category, [])
        if not f1_scores:
            print(f"{LOCOMO_CATEGORY_MAP[category]}: no samples")
            continue

        avg_f1 = statistics.mean(f1_scores)
        avg_bleu1 = statistics.mean(bleu1_scores)
        all_f1.extend(f1_scores)
        all_bleu1.extend(bleu1_scores)
        print(
            f"{LOCOMO_CATEGORY_MAP[category]}: "
            f"F1 = {avg_f1:.4f}, BLEU-1 = {avg_bleu1:.4f}, n = {len(f1_scores)}"
        )

    if all_f1:
        print("-" * 60)
        print(
            f"Overall: F1 = {statistics.mean(all_f1):.4f}, "
            f"BLEU-1 = {statistics.mean(all_bleu1):.4f}, n = {len(all_f1)}"
        )
    else:
        print("No valid LoCoMo category samples were found.")

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "all_loco_results.json"  # 使用main_loco_parse.py生成的文件
    main(file_path)
