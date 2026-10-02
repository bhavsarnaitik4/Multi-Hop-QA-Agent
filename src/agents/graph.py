import os
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from agents.state import GraphState
from agents.schemas import SearchQuery, FinalAnswer, RouteDecision
import requests
import nltk
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Download the NLTK sentence splitter model (only actually downloads the very first time)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # <-- The missing package

LOCAL_COLBERT_URL = "http://127.0.0.1:8893/api/search"

# 1. Load the variables from the .env file into the system
load_dotenv()

# 2. Optional: Verify it loaded correctly (will throw an error if missing)
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from the environment!")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)
# # 3. Initialize Gemini (LangChain automatically looks for the GOOGLE_API_KEY environment variable)
# gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash", 
#     temperature=0 
# )

ONNX_MODEL_DIR = Path("../models/deberta_onnx")
print(f"Loading ONNX runtime model from: {ONNX_MODEL_DIR}...")
selector_tokenizer = AutoTokenizer.from_pretrained(str(ONNX_MODEL_DIR))
selector_model = ORTModelForSequenceClassification.from_pretrained(str(ONNX_MODEL_DIR))
print("ONNX DeBERTa Selector loaded successfully!\n")


# --- THE PLANNER NODE (Tier 1) ---
def planner_node(state: GraphState):
    """Decomposes the question and generates the next search query."""
    current_hop = state.get("hop_count", 0) + 1
    question = state["original_question"]
    facts = state.get("selected_facts", [])
    
    # Bind the Pydantic schema to the LLM
    structured_planner = llm.with_structured_output(SearchQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert search strategist. The user is asking a complex, multi-hop question. "
                   "Based on the facts already found, generate the exact 3-5 word search query needed to find the remaining missing information."),
        ("human", "Original Question: {question}\nFacts Already Found: {facts}\n\nGenerate the next search query.")
    ])
    
    chain = prompt | structured_planner
    result = chain.invoke({"question": question, "facts": facts})
    
    print(f"\n[Planner] Hop {current_hop} Query Generated -> '{result.query}'")
    
    return {"hop_count": current_hop, "current_query": result.query}





def retrieve_node(state: dict):
    query = state["current_query"]
    print(f"[Retriever] Querying Local ColBERT for -> '{query}'")
    
    try:
        response = requests.get(
            LOCAL_COLBERT_URL, 
            params={"query": query, "k": 10}, 
            timeout=10 
        )
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print(f"Local API failed: {e}")
        return {"retrieved_paragraphs": []}

    candidates = []
    topk_docs = results.get("topk", [])
    
    for doc in topk_docs: 
        raw_text = doc.get("text", "")
        
        # Split Title and Body
        if " | " in raw_text:
            title, body = raw_text.split(" | ", 1)
        else:
            title, body = "Unknown", raw_text
        
        # Upgraded Runtime Chunking: AI-aware sentence splitting
        sentences = nltk.sent_tokenize(body) 
        
        for sent in sentences:
            if sent.strip():
                # Reattach context for DeBERTa
                candidates.append(f"{title}: {sent.strip()}")
                
    print(f"[Retriever] Chunks passed to DeBERTa: {len(candidates)}")
    return {"retrieved_paragraphs": candidates}



def selector_node(state: GraphState):
    """Production ONNX DeBERTa sentence classification."""
    question = state["original_question"]
    candidates = state.get("retrieved_paragraphs", [])
    
    print(f"[Selector] Running ONNX inference across {len(candidates)} candidate sentences...")
    
    # Pair the original question with each candidate sentence
    questions = [question] * len(candidates)
    inputs = selector_tokenizer(
        questions, 
        candidates, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    # Fast inference through ONNX Runtime
    with torch.no_grad():
        outputs = selector_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        
    # Calibrated threshold to filter false positives
    THRESHOLD = 0.70
    selected = []
    
    for sent, prob in zip(candidates, probs):
        print(f"  - Prob: {prob:.4f} | Sentence: '{sent[:60]}...'")
        if prob > THRESHOLD:
            selected.append(sent)
            
    print(f"[Selector] Filtered down to {len(selected)} verified fact(s).")
    return {"selected_facts": selected}



# --- THE SYNTHESIZER NODE (Tier 2) ---
def synthesize_node(state: dict):
    # 1. Cleanly format the list into a string
    formatted_facts = "\n".join([f"- {fact}" for fact in state.get("selected_facts", [])])
    question = state["original_question"] # <-- BUG FIX: Use original question!
    
    # 2. Define the ultra-strict prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an exact-extraction assistant. Answer the question using ONLY the provided facts. "
                   "Your final answer must be as short as humanly possible. "
                   "If the answer is a name, just output the name. "
                   "If it is a yes/no question, output ONLY 'yes' or 'no'. "
                   "Do not write full sentences.\n\n"
                   "You must format your response exactly like this:\nAnswer: \nCitations: "),
        ("human", "Question: {question}\n\nVerified Facts:\n{facts}")
    ])
    
    # 3. Create the LangChain pipeline
    chain = prompt | llm | StrOutputParser()
    
    # 4. Invoke the chain
    result = chain.invoke({
        "question": question,  
        "facts": formatted_facts
    })
    
    return {"final_answer": result, "citations": state.get("selected_facts", [])}


# --- 2. ROUTING LOGIC ---

# def sufficiency_check(state: GraphState):
#     """Deterministic Python logic to break the loop."""
#     facts = state.get("selected_facts", [])
#     hop_count = state.get("hop_count", 0)
    
#     # If we have 2 distinct facts OR we hit our 3-hop limit, stop searching.
#     if len(facts) >= 2 or hop_count >= 3:
#         print(f"[Router] Sufficient evidence found ({len(facts)} facts). Routing to Synthesis.")
#         return "synthesize"
#     else:
#         print("[Router] Evidence insufficient. Triggering next hop.")
#         return "plan_next_hop"

def sufficiency_check(state: GraphState):
    """LLM-based logic to check if all parts of the question are answered."""
    facts = state.get("selected_facts", [])
    hop_count = state.get("hop_count", 0)
    question = state["original_question"]
    
    # Fallback: don't loop forever
    if hop_count >= 3:
        print(f"[Router] Max hops reached ({hop_count}). Routing to Synthesis.")
        return "synthesize"
        
    if not facts:
        print("[Router] No facts yet. Triggering next hop.")
        return "plan_next_hop"

    # LLM Evaluation
    structured_router = llm.with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict reasoning evaluator. The user's question requires multiple pieces of evidence (e.g., finding two different people, or finding a location and then finding who lives there). Look at the current Verified Facts. Does the context contain the answer for ALL parts of the question? If even one piece is missing (like the second person's nationality), you MUST output False."),
        ("human", "Original Question: {question}\nVerified Facts: {facts}\n\nAre all parts answered?")
    ])
    
    chain = prompt | structured_router
    decision = chain.invoke({"question": question, "facts": facts})
    
    if decision.is_complete:
        print("[Router] Sufficient evidence found for all parts. Routing to Synthesis.")
        return "synthesize"
    else:
        print("[Router] Evidence incomplete (missing subjects). Triggering next hop.")
        return "plan_next_hop"
    
# --- 3. BUILD THE GRAPH ---

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("selector", selector_node)
workflow.add_node("synthesizer", synthesize_node)

# Define the Flow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "selector")

# Add Conditional Routing
workflow.add_conditional_edges(
    "selector",
    sufficiency_check,
    {
        "plan_next_hop": "planner",  # Loop back to start
        "synthesize": "synthesizer"  # Break loop, go to end
    }
)

workflow.add_edge("synthesizer", END)

# Compile
app = workflow.compile()

# --- 6. RUN LIVE END-TO-END TEST ---
if __name__ == "__main__":
    initial_state = {
        "original_question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "hop_count": 0,
        "selected_facts": []
    }
    
    print("=== RUNNING LIVE MULTI-HOP GRAPH WITH ONNX DEBERTA ===")
    final_state = app.invoke(initial_state)
    print("\n=== PIPELINE FINISHED ===")
    print(f"Total Hops: {final_state['hop_count']}")
    print(f"Selected Facts: {final_state['selected_facts']}")
    print(f"Final Answer: {final_state['final_answer']}")