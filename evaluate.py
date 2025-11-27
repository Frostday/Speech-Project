import pandas as pd
import string

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


print("="*50)
print("Pretrained OWSM CTC + Pretrained Olmo")
print("="*50)

df = pd.read_csv("baseline/baseline.csv")
print(len(df))
print(df.iloc[1])
pred_asr = df['ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Pretrained OWSM CTC + Finetuned Olmo")
print("="*50)

df = pd.read_csv("olmo_finetuning/pretrained_owsm_ctc_finetuned_olmo.csv")
print(len(df))
print(df.iloc[1])
pred_asr = df['ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['Finetuned LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Pretrained OWSM + Pretrained Olmo")
print("="*50)

df = pd.read_csv("baseline_2/baseline_2.csv")
print(df.iloc[1])
print(len(df))
df = df[df['LLM Pred'].apply(lambda x: isinstance(x, str))]
print(len(df))
pred_asr = df['ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Pretrained OWSM + Finetuned Olmo")
print("="*50)

df = pd.read_csv("olmo_finetuning/pretrained_owsm_finetuned_olmo.csv")
print(len(df))
print(df.iloc[1])
pred_asr = df['ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['Finetuned LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Finetuned OWSM CTC + Finetuned Olmo")
print("="*50)

df_original = pd.read_csv("baseline/baseline.csv")
df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/finetuned_owsm_ctc_finetuned_olmo.csv")
df = pd.merge(df_original, df, on="File-id", how="inner")
print(df.iloc[1])
print(len(df))
pred_asr = df['PREDICTED'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['Finetuned LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Finetuned OWSM + Finetuned Olmo")
print("="*50)

df_original = pd.read_csv("baseline/baseline.csv")
df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/finetuned_owsm_finetuned_olmo.csv")
df = pd.merge(df_original, df, on="File-id", how="inner")
print(df.iloc[1])
print(len(df))
pred_asr = df['PREDICTED'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['Finetuned LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
wers = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("LLM pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("LLM pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Previous text from OWSM CTC + OWSM CTC")
print("="*50)

df_original = pd.read_csv("baseline/baseline.csv")
df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_ctc_prefix/results.csv")
df = pd.merge(df_original, df, on="File-id", how="inner")
print(df.iloc[1])
print(len(df))
pred_asr = df['New ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))

print("="*50)
print("Previous text from OWSM CTC + OWSM")
print("="*50)

df_original = pd.read_csv("baseline/baseline.csv")
df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_prefix/results.csv")
df = pd.merge(df_original, df, on="File-id", how="inner")
print(df.iloc[1])
print(len(df))
pred_asr = df['New ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
ref_llm = df['LLM Reference'].tolist()
wers = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("ASR pred / ASR ref WER:", sum(wers) / len(wers))
wers = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("ASR pred / LLM ref WER:", sum(wers) / len(wers))
