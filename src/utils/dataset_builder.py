import json
import random

def balance_and_sample(input_path, output_path, target_samples):
    print(f"Balancing {input_path}...")
    positives = []
    negatives = []
    
    with open(input_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record['label'] == 1:
                positives.append(record)
            else:
                negatives.append(record)
                
    # Force 1:1 balance bounded by the target
    limit = min(len(positives), len(negatives), target_samples // 2)
    
    sampled_pos = random.sample(positives, limit)
    sampled_neg = random.sample(negatives, limit)
    
    combined = sampled_pos + sampled_neg
    random.shuffle(combined)
    
    with open(output_path, 'w') as f:
        for record in combined:
            f.write(json.dumps(record) + '\n')
            
    print(f"Saved {len(combined)} balanced records to {output_path}")

if __name__ == "__main__":
    random.seed(42)
    # Train: 50k pos + 50k neg = 100,000 pairs
    balance_and_sample("data/processed/train_sentence_labels.jsonl", "data/processed/train_balanced.jsonl", 100000)
    # Dev: 10k pos + 10k neg = 20,000 pairs
    balance_and_sample("data/processed/dev_sentence_labels.jsonl", "data/processed/dev_balanced.jsonl", 20000)
    