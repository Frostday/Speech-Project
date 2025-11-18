import json
import os

test_audio_dir = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/data/flac/eval/eval-data-release-20250327/sandi2025-challenge/data/flac/eval/01/'
test_annotations_stage_2 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-trans-ref.json'
test_annotations_stage_3 = '/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/reference-materials/annotations/eval-gec-ref.json'

# audio_file = os.path.join(test_audio_dir, os.listdir(test_audio_dir)[0])
audio_file = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025/data/flac/eval/eval-data-release-20250327/sandi2025-challenge/data/flac/eval/01/SI114J-00311-P10007.flac"
print(audio_file)
audio_id = audio_file.split('/')[-1].split('.')[0]
print(audio_id)

flag = False
with open(test_annotations_stage_2, 'r') as f:
    annotations = json.load(f)
    for annotation in annotations["files"]:
        if annotation['File-id'] == audio_id:
            print(annotation)
            flag = True
            break

if flag:
    text = ""
    for i in annotation['Transcript']:
        if 'word' not in i:
            # text += f"({i['tag']}) "
            continue
        text += i['word'] + " "
        # if 'marks' in i:
        #     text += f"({i['marks']}) "
    print(text)
else:
    print("NOT FOUND")

flag = False
with open(test_annotations_stage_3, 'r') as f:
    annotations = json.load(f)
    for annotation in annotations["files"]:
        if annotation['File-id'] == audio_id:
            print(annotation)
            flag = True
            break

if flag:
    text = ""
    for i in annotation['Transcript']:
        if 'word' not in i:
            # text += f"({i['tag']}) "
            continue
        text += i['word'] + " "
        # if 'marks' in i:
        #     text += f"({i['marks']}) "
    print(text)
else:
    print("NOT FOUND")
