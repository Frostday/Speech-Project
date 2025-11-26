import torch
import datasets
import espnet
import espnetez as ez
import numpy as np
import soundfile as sf
import json
import glob
import os
import pandas as pd
from tqdm import tqdm

from espnet2.bin.s2t_inference import Speech2Text

print("Successfully imported espnet ez")
print("ESPnet version: ", espnet.__version__)

FINETUNE_MODEL = "espnet/owsm_v4_medium_1B"
owsm_language = "<eng>"

MODEL_CHECKPOINT = "/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/exp/finetune/5epoch.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base pretrained model (for tokenizer, config, and as a backbone for loading ckpt)
pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    device=DEVICE,
    dtype="float32",
    lang_sym=owsm_language,
    beam_size=10,
)

pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))


base_data_path = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025"
eval_flac_path = f"{base_data_path}/data/flac/eval/eval-data-release-20250327/sandi2025-challenge/data/flac/eval"
eval_annotations = f"{base_data_path}/reference-materials/annotations/eval-gec-ref.json"


def load_annotations(annotation_path):
    """Load annotations from JSON file."""
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    # Create a dictionary mapping File-id to transcript text
    annotation_dict = {}
    for annotation in annotations.get("files", []):
        file_id = annotation.get("File-id", "")
        transcript = ""
        for item in annotation.get("Transcript", []):
            if "word" in item:
                transcript += item["word"] + " "
        transcript = transcript.strip()
        annotation_dict[file_id] = transcript
    return annotation_dict

def create_dataset_from_flac(flac_dir, annotation_dict):
    """Create a dataset from FLAC files and annotations.
    Only includes audio files that have corresponding annotations.
    """
    # Find all FLAC files recursively
    flac_files = glob.glob(f"{flac_dir}/**/*.flac", recursive=True)
    
    dataset_items = []
    skipped_no_annotation = 0
    skipped_load_error = 0
    
    for flac_path in flac_files:
        file_id = os.path.splitext(os.path.basename(flac_path))[0]
        
        # Get transcript from annotations - skip if not available
        transcript = annotation_dict.get(file_id, "")
        if not transcript:
            skipped_no_annotation += 1
            continue  # Skip files without annotations
        
        # Load audio
        try:
            wav, sr = sf.read(flac_path, dtype="float32")
            dataset_items.append({
                "file_id": file_id,
                "speech": wav,
                "transcription": transcript,
                "sampling_rate": sr
            })
        except Exception as e:
            skipped_load_error += 1
            print(f"Error loading {flac_path}: {e}")
            continue
    
    # Report statistics
    total_files = len(flac_files)
    used_files = len(dataset_items)
    print(f"  Total FLAC files found: {total_files}")
    print(f"  Files with annotations (used): {used_files}")
    print(f"  Files skipped (no annotation): {skipped_no_annotation}")
    print(f"  Files skipped (load error): {skipped_load_error}")
    
    return dataset_items


print("Loading annotations...")
eval_ann_dict = load_annotations(eval_annotations)
print(f"Loaded {len(eval_ann_dict)} eval annotations")

print("Loading eval audio files...")
eval_items = create_dataset_from_flac(eval_flac_path, eval_ann_dict)
print(f"Loaded {len(eval_items)} eval audio files")

# Convert to HuggingFace datasets format for espnetez
def items_to_dataset(items):
    """Convert list of items to HuggingFace dataset format."""
    data_dict = {
        "speech": [item["speech"] for item in items],
        "transcription": [item["transcription"] for item in items],
        "file_id": [item["file_id"] for item in items],
    }
    return datasets.Dataset.from_dict(data_dict)

eval_hf = items_to_dataset(eval_items)


test_data_info = {
    "speech": lambda d: d["speech"].astype(np.float32)
    if isinstance(d["speech"], np.ndarray) else np.array(d["speech"], dtype=np.float32),
    "text": lambda d: tokenize(f"{owsm_language}<asr><notimestamps> {d['transcription']}"),
    "text_ctc": lambda d: tokenize(d['transcription']),
    "text_prev": lambda d: tokenize("<na>"),
    "text_raw": lambda d: d['transcription'],
}

test_dataset = ez.dataset.ESPnetEZDataset(eval_hf, data_info=test_data_info)

# -------------------------------------------------------------------------
# Load fine-tuned weights into the seq2seq model
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("Loading fine-tuned checkpoint...")
print("="*80)

if not os.path.isfile(MODEL_CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found at {MODEL_CHECKPOINT}")

state_dict = torch.load(MODEL_CHECKPOINT, map_location=pretrained_model.device)
pretrained_model.s2t_model.load_state_dict(state_dict, strict=False)
pretrained_model.s2t_model.to(pretrained_model.device)
pretrained_model.s2t_model.float().eval()

print(f"Model device: {pretrained_model.device}")
print(f"Total test samples: {len(eval_hf)}")

# -------------------------------------------------------------------------
# Inference loop
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("Generating transcripts for test/eval data...")
print("="*80)

predictions_data = []

for i in tqdm(range(len(eval_hf))):
    uid, sample = test_dataset.__getitem__(i)
    try:
        # Speech2Text returns a list of n-best results.
        # For seq2seq OWSM, each n-best item is usually: (text, token, token_int, hyp)
        nbest = pretrained_model(sample["speech"])
        predicted_text = nbest[0][3]

        reference_text = sample["text_raw"]
        file_id = eval_hf[i]["file_id"]
        
        predictions_data.append({
            "File-id": file_id,
            "PREDICTED": predicted_text,
            # "REFERENCE": reference_text,
        })

        # Optional: light logging
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(eval_hf)} samples...")

    except Exception as e:
        print(f"Error processing sample {i} (file_id: {eval_hf[i]['file_id']}): {e}")
        predictions_data.append({
            "File-id": eval_hf[i]["file_id"],
            "PREDICTED": f"ERROR: {str(e)}",
            # "REFERENCE": eval_hf[i]["transcription"] if i < len(eval_hf) else "N/A",
        })
        continue

# -------------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------------
predictions_df = pd.DataFrame(predictions_data)
output_csv = "./test_predictions.csv"
predictions_df.to_csv(output_csv, index=False)

print("\n" + "="*80)
print(f"Finished generating transcripts. Saved {len(predictions_data)} predictions to {output_csv}")
print("="*80)
