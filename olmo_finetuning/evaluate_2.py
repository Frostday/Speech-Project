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


# CSV 1

df_original = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/baseline/baseline.csv")
print(len(df_original))

df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/finetuned_owsm_finetuned_olmo.csv")
print(len(df))

df = pd.merge(df_original, df, on="File-id", how="inner")
print(len(df))

pred_asr = df["Finetuned LLM Pred"].tolist()
ref_asr = df["ASR Reference"].tolist()
ref_llm = df["LLM Reference"].tolist()
print("Finetuned Olmo on finetuned OWSM ASR")

wer_vals = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("WER (ASR):", sum(wer_vals) / len(wer_vals))

wer_vals = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("WER (Corrected):", sum(wer_vals) / len(wer_vals))

# CSV 2

df_original = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/baseline/baseline.csv")
print(len(df_original))

df = pd.read_csv("/ocean/projects/cis250187p/dgarg2/Speech-Project/olmo_finetuning/finetuned_owsm_ctc_finetuned_olmo.csv")
print(len(df))

df = pd.merge(df_original, df, on="File-id", how="inner")
print(len(df))

pred_asr = df["Finetuned LLM Pred"].tolist()
ref_asr = df["ASR Reference"].tolist()
ref_llm = df["LLM Reference"].tolist()
print("Finetuned Olmo on finetuned OWSM CTC ASR")

wer_vals = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
print("WER (ASR):", sum(wer_vals) / len(wer_vals))

wer_vals = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
print("WER (Corrected):", sum(wer_vals) / len(wer_vals))
