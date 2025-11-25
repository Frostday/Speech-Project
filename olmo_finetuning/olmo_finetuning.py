import os
import pandas as pd
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ----------------------------
# Config (edit these)
# ----------------------------

MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"
TRAIN_CSV = "train_data.csv"  # columns: File-id,ASR,Corrected
OUTPUT_DIR = "./olmo_asr_correction_lora"

BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
GRADIENT_ACCUMULATION_STEPS = 8


# ----------------------------
# Dataset
# ----------------------------

class ASRCorrectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def build_message(self, asr: str, corrected: str) -> List[Dict[str, str]]:
        """
        Format exactly as you requested:

        message = [
            {"role": "user", "content": f"Remove grammatical errors in ASR transcript WITHOUT changing content: {asr}"},
            {"role": "assistant", "content": "Here's the corrected version: {corrected}"}
        ]
        """
        messages = [
            {
                "role": "user",
                "content": f"Remove grammatical errors in ASR transcript WITHOUT changing content: {asr}",
            },
            {
                "role": "assistant",
                "content": f"Here's the corrected version: {corrected}",
            },
        ]
        return messages

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages list into a single prompt string that OLMo can train on.
        """
        text_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "user":
                text_parts.append(f"<|user|>: {content}")
            elif role == "assistant":
                text_parts.append(f"<|assistant|>: {content}")
            else:
                text_parts.append(f"<|{role}|>: {content}")
        # Add end-of-sequence token to mark completion
        prompt = "\n".join(text_parts) + self.tokenizer.eos_token
        return prompt

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        asr = str(row["ASR"])
        corrected = str(row["Corrected"])

        messages = self.build_message(asr, corrected)
        prompt = self.messages_to_prompt(messages)

        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Trainer expects 1D tensors, not batch dimension
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # For causal LM, labels are usually same as input_ids
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ----------------------------
# Main
# ----------------------------

def main():
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # Ensure EOS & PAD are set (OLMo is causal LM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 2. Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load data
    df = pd.read_csv(TRAIN_CSV)
    # basic sanity-check
    required_cols = {"ASR", "Corrected"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    dataset = ASRCorrectionDataset(df, tokenizer, max_length=MAX_LENGTH)

    # 4. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",  # change if you add a val set
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        fp16=torch.cuda.is_available(),
        bf16=False,  # set True if your GPU supports bf16 and you want it
        report_to="none",
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 7. Train
    trainer.train()

    # 8. Save adapter + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
