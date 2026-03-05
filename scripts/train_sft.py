# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45",
#     "datasets>=3.0",
#     "torch>=2.4",
#     "accelerate>=1.0",
#     "trackio",
# ]
# ///
"""SFT Training for elias-memory agent on HuggingFace Jobs.

Trains Qwen2.5-3B on memory operation traces using LoRA.
Dataset: erik1988/elias-memory-traces-v1 (24 traces)
Output: erik1988/elias-memory-agent-v1
"""

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

DATASET_ID = "erik1988/elias-memory-traces-v1"
BASE_MODEL = "Qwen/Qwen2.5-3B"
OUTPUT_MODEL = "erik1988/elias-memory-agent-v1"

# Load dataset
dataset = load_dataset(DATASET_ID, data_files="train.jsonl", split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# LoRA config (efficient fine-tuning)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Training config
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=5,
    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    gradient_checkpointing=True,
    bf16=True,
    report_to="trackio",
    project="elias-memory",
    run_name="sft-v1-qwen25-3b",
)

# Train
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.push_to_hub()
print(f"Model pushed to https://huggingface.co/{OUTPUT_MODEL}")
