from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import string
from tqdm import tqdm

base_model_name = "allenai/OLMo-2-0425-1B-Instruct"
adapter_dir = "./olmo_asr_correction_lora/checkpoint-600"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading model")
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
print("Loaded model")

def correct_asr(asr: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Remove grammatical errors in ASR transcript WITHOUT changing content: {asr}",
        }
    ]
    text = "\n".join([f"<|{m['role']}|>: {m['content']}" for m in messages]) + "\n<|assistant|>:"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # crude way: take everything after last "<|assistant|>:"
    if "<|assistant|>:" in full:
        full = full.split("<|assistant|>:")[-1].strip()
    return full


print(correct_asr("We can see that factories and vehicles make pollution. This this pollution make acid particles because these acid particles are are in the smoke of the factories and the vehicles. and after these acid particles are carried by the by wind. and in all the country, these acid particles fall in as dusts, but others go in the cloud, and when it's when there is rain, these acid particles mix with the cloud water, are falling and hurts the environment and the planet."))

df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/test_predictions.csv")
asr_text = df['PREDICTED'].tolist()
pred_llm = []
for i in tqdm(asr_text):
    result = correct_asr(i)
    if result.startswith("Here's the corrected version:"):
        result = result[len("Here's the corrected version:"):]
    pred_llm.append(result)
df['Finetuned LLM Pred'] = pred_llm
df.to_csv("finetuned_owsm_finetuned_olmo.csv", index=False)

df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_ctc_finetuning/test_predictions.csv")
# DATA CLEANING for OWSM CTC
df["PREDICTED"] = df["PREDICTED"].str.split(n=1).str[1].fillna("")
asr_text = df['PREDICTED'].tolist()
pred_llm = []
for i in tqdm(asr_text):
    result = correct_asr(i)
    if result.startswith("Here's the corrected version:"):
        result = result[len("Here's the corrected version:"):]
    pred_llm.append(result)
df['Finetuned LLM Pred'] = pred_llm
df.to_csv("finetuned_owsm_ctc_finetuned_olmo.csv", index=False)
