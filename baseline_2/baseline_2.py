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

PRETRAINED_MODEL="espnet/owsm_v4_medium_1B"
owsm_language="<eng>"

pretrained_model = Speech2Text.from_pretrained(
    PRETRAINED_MODEL,
    device="cuda",
    dtype="bfloat16",
    lang_sym=owsm_language,
    beam_size=10,
)

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B-Instruct", dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")

if os.path.exists("baseline_2.csv"):
    df = pd.read_csv("baseline_2.csv")
    ids_done = df["File-id"].tolist()
    PATHS = [PATH for PATH in PATHS if PATH.split('/')[-1].split('.')[0] not in ids_done]

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

    # text_prev = ""
    # pred = pretrained_model(data["speech"], text_prev)
    pred = pretrained_model(data["speech"])
    pred = pred[0][3]
    # print('PREDICTED: ' + pred)
    # print('ASR REFERENCE: ' + ref)
    # print('REFERENCE: ' + data["text"])
    # print("-"*100)
    data["asr_pred"] = pred

    message = [
        {"role": "user", "content": f"Remove grammatical errors in ASR transcript WITHOUT changing content: {pred}"},
        {"role": "assistant", "content": "Here's the corrected version: "}
    ]
    inputs = tokenizer.apply_chat_template(message, return_tensors='pt', return_token_type_ids=False)
    inputs, olmo = inputs.to('cuda'), olmo.to('cuda')
    response = olmo.generate(inputs, max_new_tokens=256, do_sample=False)
    pred_withllm_raw = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    # print(pred_withllm_raw)
    # print("-"*100)
    pred = pred_withllm_raw.split("Here's the corrected version: ")[1]
    # print('LLM Output:', pred)
    # print("-"*100)
    data["llm_pred"] = pred

    final_data.append({
        "File-id": PATH.split('/')[-1].split('.')[0],
        "ASR Pred": data["asr_pred"],
        "LLM Pred": data["llm_pred"],
        "ASR Reference": data["asr_text"],
        "LLM Reference": data["corrected_text"]
    })
    if os.path.exists("baseline_2.csv"):
        df = pd.read_csv("baseline_2.csv")
        df = pd.concat([df, pd.DataFrame([final_data[-1]])], ignore_index=True)
        df.to_csv("baseline_2.csv", index=False)
    else:
        df = pd.DataFrame(final_data)
        df.to_csv("baseline_2.csv", index=False)

    gc.collect()
    torch.cuda.empty_cache()
