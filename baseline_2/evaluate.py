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


df = pd.read_csv("baseline_2.csv")
print(len(df))
df = df[df['LLM Pred'].apply(lambda x: isinstance(x, str))]
print(len(df))

pred_asr = df['ASR Pred'].tolist()
ref_asr = df['ASR Reference'].tolist()
pred_llm = df['LLM Pred'].tolist()
ref_llm = df['LLM Reference'].tolist()

wers_1 = [wer(ref_asr[i], pred_asr[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
wers_2 = [wer(ref_asr[i], pred_llm[i]) for i in range(len(ref_asr)) if ref_asr[i] != "NOT FOUND"]
wers_3 = [wer(ref_llm[i], pred_asr[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
wers_4 = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]

print("ASR ref ASR pred WER:", sum(wers_1) / len(wers_1))
print("ASR ref LLM pred WER:", sum(wers_2) / len(wers_2))
print("LLM ref ASR pred WER:", sum(wers_3) / len(wers_3))
print("LLM ref LLM pred WER:", sum(wers_4) / len(wers_4))

# cleaned_pred_llm = []
# for text in pred_llm:
#     if "revised version:" in text.lower():
#         text = text.split("revised version:")[-1].strip()
#     if "Changes made:" in text:
#         text = text.split("Changes made:")[0].strip()
#     cleaned_pred_llm.append(text)
# pred_llm = cleaned_pred_llm
# wer = [wer(ref_llm[i], pred_llm[i]) for i in range(len(ref_llm)) if ref_llm[i] != "NOT FOUND"]
# for i in range(len(ref_llm)):
#     if ref_llm[i] != "NOT FOUND" and (len(ref_llm[i])-len(pred_llm[i]) >= 30 or len(ref_llm[i])-len(pred_llm[i]) <= -30 or "\n" in pred_llm[i]):
#         # print("REF:", ref_llm[i])
#         # print("PRED:", pred_llm[i])
#         pred_llm[i] = pred_asr[i]
# print("Cleaned up LLM responses:", sum(wer) / len(wer))
