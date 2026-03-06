# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers>=4.45", "torch>=2.4", "peft>=0.7.0", "huggingface_hub"]
# ///
"""Merge LoRA adapter with base model and push to Hub for GGUF conversion."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-3B"
ADAPTER = "erik1988/elias-memory-agent-v1"
OUTPUT = "erik1988/elias-memory-agent-v1-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("./merged")
tokenizer.save_pretrained("./merged")

print("Pushing merged model to Hub...")
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(OUTPUT, exist_ok=True)
api.upload_folder(folder_path="./merged", repo_id=OUTPUT)
print(f"Merged model pushed to https://huggingface.co/{OUTPUT}")
print("GGUF conversion can be done on Jetson with: llama-quantize")
