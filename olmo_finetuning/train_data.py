import numpy as np
import pandas as pd
import json
import os
import glob
from tqdm import tqdm

train_annotations_stage_2 = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/train-trans-ref.json"
train_annotations_stage_3 = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/train-gec-ref.json"
# train_annotations_stage_2 = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/dev-trans-ref.json"
# train_annotations_stage_3 = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/dev-gec-ref.json"

data = []
with open(train_annotations_stage_3, 'r') as f3:
    with open(train_annotations_stage_2, 'r') as f2:
        annotations_2 = json.load(f2)
        annotations_3 = json.load(f3)
        for annotation in tqdm(annotations_2["files"]):
            file_id = annotation['File-id']
            correct_annotation = [a for a in annotations_3["files"] if a['File-id']==file_id]
            if len(correct_annotation)==0:
                continue
            correct_annotation = correct_annotation[0]
            asr = ""
            for i in annotation['Transcript']:
                if 'word' not in i:
                    continue
                asr += i['word'] + " "
            corr = ""
            for i in correct_annotation['Transcript']:
                if 'word' not in i:
                    continue
                corr += i['word'] + " "
            data.append({
                "File-id": file_id,
                "ASR": asr,
                "Corrected": corr
            })

df = pd.DataFrame(data)
df.to_csv("train_data.csv", index=False)
