import sys
import os
import time

# 1. Get the absolute path to the 'utils' folder
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the path to the parent root folder
parent_dir = os.path.dirname(current_dir)

# 3. Add the root folder to Python's path
sys.path.append(parent_dir)

# 4. Now you can successfully import from the agent folder
import json
import re
import string
from collections import Counter
from tqdm import tqdm
from agents.graph import app  # Imports your compiled LangGraph

# --- Official HotpotQA Metric Functions ---
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

# --- Main Evaluation Loop ---
def run_evaluation(dataset_path, output_path, limit=50):
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
        
    # Optional: limit the number of questions for initial testing
    if limit:
        dataset = dataset[:limit]

    results = []
    total_em = 0
    total_f1 = 0

    print(f"\nStarting evaluation on {len(dataset)} questions...\n")
    
    for item in tqdm(dataset, desc="Evaluating"):
        question = item['question']
        ground_truth = item['answer']
        
        initial_state = {
            "question": question,
            "original_question": question,
            "current_query": question,
            "hop_count": 0,
            "retrieved_paragraphs": [],
            "selected_facts": [],
            "final_answer": "",
            "evaluation": ""
        }
        
        try:
            # Run your local Fullwiki pipeline
            final_state = app.invoke(initial_state)
            
            # Extract the raw answer string from your synthesizer's format
            raw_output = final_state.get('final_answer', '')
            
            # Basic parsing to extract just the answer text (stripping 'Answer:' and citations)
            predicted_answer = raw_output
            if "Answer:" in raw_output:
                predicted_answer = raw_output.split("Answer:")[1].split("Citations:")[0].strip()

            em = exact_match_score(predicted_answer, ground_truth)
            f1, prec, rec = f1_score(predicted_answer, ground_truth)
            
            total_em += int(em)
            total_f1 += f1
            
            results.append({
                "id": item.get("_id", "unknown"),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted_answer,
                "hops": final_state.get('hop_count', 0),
                "em": bool(em),
                "f1": round(f1, 4)
            })
            
        except Exception as e:
            print(f"\nError processing question: {question}\n{e}")
            results.append({"question": question, "error": str(e)})
        

        # Save checkpoints after every question in case of failure
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        # In utils/test.py at the bottom of the question loop:
        time.sleep(2)  # 2 seconds is ample for Groq

    # Calculate final averages
    avg_em = (total_em / len(dataset)) * 100
    avg_f1 = (total_f1 / len(dataset)) * 100
    
    print("\n" + "="*50)
    print("🏆 EVALUATION COMPLETE")
    print(f"Total Questions: {len(dataset)}")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score: {avg_f1:.2f}%")
    print(f"Detailed results saved to {output_path}")
    print("="*50)

if __name__ == "__main__":
    # Point this to your downloaded HotpotQA dev set JSON file
    DATASET_FILE = "../../data/raw/hotpot_dev_fullwiki_v1.json" 
    OUTPUT_FILE = "evaluation_results.json"
    
    # We limit to 50 initially because LLM API calls take time. 
    # Change to limit=None to run the entire dev set.
    run_evaluation(DATASET_FILE, OUTPUT_FILE, limit=50)