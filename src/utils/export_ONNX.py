import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from optimum.onnxruntime import ORTModelForSequenceClassification

def merge_and_export():
    # 1. Paths
    adapter_path = "../models/hotpotqa-deberta-adapter" # Adjust this if your path is different
    merged_path = "../models/merged_pytorch_model"
    onnx_output_path = "../models/deberta_onnx"
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large", 
        num_labels=2, 
        torch_dtype=torch.float32
    )
    
    print("Fusing LoRA adapter into base weights...")
    # Load the LoRA adapter on top of the base model
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Permanently merge the adapter weights into the base layers
    merged_model = peft_model.merge_and_unload()
    
    # Save the standard PyTorch model temporarily
    print(f"Saving merged PyTorch model to {merged_path}...")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    print("Exporting to ONNX Runtime via Optimum...")
    # Loading it via ORTModelForSequenceClassification with export=True triggers the ONNX conversion
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        merged_path,
        export=True
    )
    
    print(f"Saving final ONNX graph to {onnx_output_path}...")
    ort_model.save_pretrained(onnx_output_path)
    tokenizer.save_pretrained(onnx_output_path)
    
    print("\nSUCCESS: Your DeBERTa model is now a fully optimized ONNX graph!")

if __name__ == "__main__":
    merge_and_export()