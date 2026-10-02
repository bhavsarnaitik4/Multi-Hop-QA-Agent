import json
from datasets import load_dataset

print("Downloading HotpotQA Fullwiki Dev Set from Hugging Face...")

# Load the validation split (which is the dev set) of the Fullwiki config
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

hotpot_data = []
for item in dataset:
    # We map the Hugging Face columns to the exact keys evaluate.py expects
    hotpot_data.append({
        "_id": item["id"],
        "question": item["question"],
        "answer": item["answer"]
    })

# Dump it into the local JSON file
with open("../../data/raw/hotpot_dev_fullwiki_v1.json", "w") as f:
    json.dump(hotpot_data, f, indent=4)

print(f"✅ Successfully saved {len(hotpot_data)} questions to hotpot_dev_fullwiki_v1.json!")