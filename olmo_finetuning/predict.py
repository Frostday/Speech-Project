from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import pandas as pd
import string

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

def normalize(text):
    """Lowercase and remove punctuation."""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

def wer(ref, hyp):
    """
    Compute Word Error Rate (WER) ignoring punctuation.
    WER = (S + D + I) / N
    """
    # Normalize
    ref = normalize(ref).split()
    hyp = normalize(hyp).split()
    # print("-"*100)
    # print(ref)
    # print(hyp)

    N = len(ref)

    # Initialize DP matrix
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    # Compute DP (Levenshtein)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # deletion
                    dp[i][j-1],     # insertion
                    dp[i-1][j-1]    # substitution
                )

    wer_value = dp[-1][-1] / N if N > 0 else 0.0
    return wer_value


print(correct_asr("We can see that factories and vehicles make pollution. This this pollution make acid particles because these acid particles are are in the smoke of the factories and the vehicles. and after these acid particles are carried by the by wind. and in all the country, these acid particles fall in as dusts, but others go in the cloud, and when it's when there is rain, these acid particles mix with the cloud water, are falling and hurts the environment and the planet."))

df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/baseline/baseline.csv")
asr_text = df['ASR Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
pred_llm = []
for i in asr_text:
    result = correct_asr(i)
    if result.startswith("Here's the corrected version:"):
        result = result[len("Here's the corrected version:"):]
    pred_llm.append(result)
df['Finetuned LLM Pred'] = pred_llm
df.to_csv("finetuned_olmo.csv", index=False)
wer = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("WER:", sum(wer) / len(wer))
