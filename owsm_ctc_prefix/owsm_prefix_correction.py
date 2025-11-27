import torch
import datasets
import numpy as np
import pandas as pd
import soundfile as sf
from espnet2.bin.s2t_inference import Speech2Text
from transformers import AutoModelForCausalLM, AutoTokenizer
from espnet2.bin.s2t_inference_ctc import Speech2Text as CTCInfer
import json
import os
import glob
from tqdm import tqdm
import gc

test_annotations_stage_2 = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-trans-ref.json"
test_annotations_stage_3 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-gec-ref.json'
PATHS = glob.glob("/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/data/flac/eval/eval-data-release-20250327/sandi2025-challenge/data/flac/eval/01/*.flac")

PRETRAINED_MODEL="espnet/owsm_ctc_v4_1B"
owsm_language="<eng>"

pretrained_model = CTCInfer.from_pretrained(
    PRETRAINED_MODEL,
    device="cuda",
    dtype="bfloat16",
    lang_sym=owsm_language,
    task_sym='<asr>'
)

df = pd.read_csv("../baseline/baseline.csv")

final_data = []
for PATH in tqdm(PATHS):
    flag = False
    with open(test_annotations_stage_2, 'r') as f:
        annotations = json.load(f)
        for annotation in annotations["files"]:
            if annotation['File-id'] == PATH.split('/')[-1].split('.')[0]:
                flag = True
                break
    if flag:
        ref = ""
        for i in annotation['Transcript']:
            if 'word' not in i:
                continue
            ref += i['word'] + " "
    else:
        ref = "NOT FOUND"

    flag = False
    with open(test_annotations_stage_3, 'r') as f:
        annotations = json.load(f)
        for annotation in annotations["files"]:
            if annotation['File-id'] == PATH.split('/')[-1].split('.')[0]:
                flag = True
                break
    if flag:
        text = ""
        for i in annotation['Transcript']:
            if 'word' not in i:
                continue
            text += i['word'] + " "
    else:
        text = "NOT FOUND"

    wav, sr = sf.read(PATH, dtype="float32")
    data = {
        "speech": wav, 
        "asr_text": ref,
        "corrected_text": text
    }

    file_id = PATH.split('/')[-1].split('.')[0]
    filtered_df = df[df["File-id"]==file_id]
    text_prev = "Correct this transcript: " + filtered_df["ASR Pred"].values[0] if len(filtered_df) > 0 else ""
    print(text_prev)
    pred = pretrained_model(data["speech"], text_prev)
    pred = pred[0][3]
    data["asr_pred"] = pred

    final_data.append({
        "File-id": file_id,
        "New ASR Pred": data["asr_pred"],
    })

df_new = pd.DataFrame(final_data)
df_new.to_csv("results.csv", index=False)
