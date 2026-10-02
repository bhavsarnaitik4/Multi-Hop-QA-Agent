
import json
import os
from datasets import load_dataset
from tqdm import tqdm

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

def process_hf_split(split_name, output_jsonl):
    print(f"Downloading '{split_name}' split from Hugging Face...")
    
    # The 'distractor' configuration provides the 10 candidate paragraphs (2 gold + 8 decoys)
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
    
    extracted_count = 0
    positive_class = 0
    
    with open(output_jsonl, 'w') as f_out:
        for record in tqdm(dataset):
            q_id = record['id']
            question = record['question']
            
            # Reconstruct the fast-lookup set for gold supporting facts[cite: 1]
            # HF format provides dicts of lists: {'title': [...], 'sent_id': [...]}
            gold_titles = record['supporting_facts']['title']
            gold_sent_ids = record['supporting_facts']['sent_id']
            gold_facts = {f"{t}_{s}" for t, s in zip(gold_titles, gold_sent_ids)}
            
            # Iterate through the provided paragraphs[cite: 1]
            ctx_titles = record['context']['title']
            ctx_sentences = record['context']['sentences']
            
            for title, sentences in zip(ctx_titles, ctx_sentences):
                for s_idx, sentence in enumerate(sentences):
                    is_supporting = int(f"{title}_{s_idx}" in gold_facts)
                    
                    row = {
                        "q_id": q_id,
                        "question": question,
                        "context_title": title,
                        "sentence": sentence.strip(),
                        "label": is_supporting
                    }
                    f_out.write(json.dumps(row) + '\n')
                    
                    extracted_count += 1
                    if is_supporting:
                        positive_class += 1
                        
    print(f"Done processing {split_name}. Extracted {extracted_count} sentences.")
    print(f"Class balance: {positive_class} positive ({positive_class/extracted_count:.1%}) / {extracted_count - positive_class} negative")

if __name__ == "__main__":
    process_hf_split("train", "data/processed/train_sentence_labels.jsonl")
    process_hf_split("validation", "data/processed/dev_sentence_labels.jsonl")