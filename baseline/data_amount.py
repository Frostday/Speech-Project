import json
import os

train_annotations_stage_2 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/train-trans-ref.json'
train_annotations_stage_3 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/train-gec-ref.json'

dev_annotations_stage_2 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/dev-trans-ref.json'
dev_annotations_stage_3 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/dev-gec-ref.json'

test_annotations_stage_2 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-trans-ref.json'
test_annotations_stage_3 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-gec-ref.json'

with open(train_annotations_stage_2, 'r') as f:
    annotations = json.load(f)
    print("Train stage 2:", len(annotations["files"]))
with open(train_annotations_stage_3, 'r') as f:
    annotations = json.load(f)
    print("Train stage 3:", len(annotations["files"]))

with open(dev_annotations_stage_2, 'r') as f:
    annotations = json.load(f)
    print("Dev stage 2:", len(annotations["files"]))
with open(dev_annotations_stage_3, 'r') as f:
    annotations = json.load(f)
    print("Dev stage 3:", len(annotations["files"]))

with open(test_annotations_stage_2, 'r') as f:
    annotations = json.load(f)
    print("Test stage 2:", len(annotations["files"]))
with open(test_annotations_stage_3, 'r') as f:
    annotations = json.load(f)
    print("Test stage 3:", len(annotations["files"]))
    
