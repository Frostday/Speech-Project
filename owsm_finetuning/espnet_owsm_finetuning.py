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
from espnet2.bin.s2t_inference_ctc import Speech2Text as CTCInfer

print("Successfully imported espnet ez")
print("ESPnet version: ", espnet.__version__)

FINETUNE_MODEL = "espnet/owsm_ctc_v4_1B"
owsm_language = "<eng>"
pretrained_model = CTCInfer.from_pretrained(
    FINETUNE_MODEL,
    device="cuda",
    dtype="bfloat16",
    lang_sym=owsm_language,
    task_sym='<asr>'
)

# os.makedirs('models', exist_ok=True)
# torch.save(pretrained_model.s2t_model.state_dict(), 'models/original.pth')
pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter

def tokenize(text):
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

# Paths for data and annotations
base_data_path = "/ocean/projects/cis250187p/shared/speech_and_improve_corpus/sandi-corpus-2025"
dev_flac_path = f"{base_data_path}/data/flac/dev/extracted/data/flac/dev"
eval_flac_path = f"{base_data_path}/data/flac/eval/eval-data-release-20250327/sandi2025-challenge/data/flac/eval"

dev_annotations = f"{base_data_path}/reference-materials/annotations/dev-gec-ref.json"
eval_annotations = f"{base_data_path}/reference-materials/annotations/eval-gec-ref.json"

# Train data paths (commented out - uncomment to use train data instead of dev)
# train_flac_path = f"{base_data_path}/data/flac/train/extracted/data/flac/train"
# train_annotations = f"{base_data_path}/reference-materials/annotations/train-gec-ref.json"

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

# Load annotations
print("Loading annotations...")
dev_ann_dict = load_annotations(dev_annotations)
eval_ann_dict = load_annotations(eval_annotations)

# Train data loading (commented out - uncomment to use train data instead of dev)
# train_ann_dict = load_annotations(train_annotations)

print(f"Loaded {len(dev_ann_dict)} dev annotations")
print(f"Loaded {len(eval_ann_dict)} eval annotations")
# print(f"Loaded {len(train_ann_dict)} train annotations")

# Create datasets from FLAC files
print("Loading audio files...")
dev_items = create_dataset_from_flac(dev_flac_path, dev_ann_dict)
eval_items = create_dataset_from_flac(eval_flac_path, eval_ann_dict)

# Train data loading (commented out - uncomment to use train data instead of dev)
# train_items = create_dataset_from_flac(train_flac_path, train_ann_dict)

print(f"Loaded {len(dev_items)} dev audio files")
print(f"Loaded {len(eval_items)} eval audio files")
# print(f"Loaded {len(train_items)} train audio files")

# Convert to HuggingFace datasets format for espnetez
def items_to_dataset(items):
    """Convert list of items to HuggingFace dataset format."""
    data_dict = {
        "speech": [item["speech"] for item in items],
        "transcription": [item["transcription"] for item in items],
        "file_id": [item["file_id"] for item in items],
    }
    return datasets.Dataset.from_dict(data_dict)

# Use dev data for training - split into train and validation
dev_hf = items_to_dataset(dev_items)
eval_hf = items_to_dataset(eval_items)

# Split dev data into train (80%) and validation (20%)
dev_hf = dev_hf.shuffle(seed=42)
split = dev_hf.train_test_split(test_size=0.2, seed=42)
train_hf = split["train"]
valid_hf = split["test"]

# Alternative: Use train data instead of dev (commented out)
# train_hf = items_to_dataset(train_items)
# train_hf = train_hf.shuffle(seed=42)
# # Split train data into train (80%) and validation (20%)
# split = train_hf.train_test_split(test_size=0.2, seed=42)
# train_hf = split["train"]
# valid_hf = split["test"]
# # Keep dev for separate validation if needed
# dev_hf = items_to_dataset(dev_items)

print(f"Train dataset size: {len(train_hf)}, first item transcription: {train_hf[0]['transcription']}")
print(f"Validation dataset size: {len(valid_hf)}")

# Print example data
print("\n" + "="*80)
print("Example data from training set:")
print("="*80)
example_idx = 0
example_file_id = train_hf[example_idx]['file_id']
example_transcript = train_hf[example_idx]['transcription']
print(f"File ID: {example_file_id}")
print(f"Transcript: {example_transcript}")
print("="*80 + "\n")

# Data info for training and validation
data_info = {
    "prefix": lambda d, lang_task_tokens=tokenize(f"{owsm_language}<asr>")[:2].copy(): lang_task_tokens.copy(),
    "speech": lambda d: d["speech"].astype(np.float32) if isinstance(d["speech"], np.ndarray) else np.array(d["speech"], dtype=np.float32),
    "text": lambda d: tokenize(f"{owsm_language}<asr><notimestamps> {d['transcription']}"),
    "text_ctc": lambda d: tokenize(d['transcription']),
    "text_prev": lambda d: tokenize("<na>"),
}

# Data info for test (includes raw text)
test_data_info = {
    "prefix": lambda d, lang_task_tokens=tokenize(f"{owsm_language}<asr>")[:2].copy(): lang_task_tokens.copy(),
    "speech": lambda d: d["speech"].astype(np.float32) if isinstance(d["speech"], np.ndarray) else np.array(d["speech"], dtype=np.float32),
    "text": lambda d: tokenize(f"{owsm_language}<asr><notimestamps> {d['transcription']}"),
    "text_ctc": lambda d: tokenize(d['transcription']),
    "text_prev": lambda d: tokenize("<na>"),
    "text_raw": lambda d: d['transcription'],
}

# Create ESPnetEZ datasets
# Use dev data (split) for training and validation
train_dataset = ez.dataset.ESPnetEZDataset(train_hf, data_info=data_info)
valid_dataset = ez.dataset.ESPnetEZDataset(valid_hf, data_info=data_info)
# Eval dataset is only for inference/transcript generation
test_dataset = ez.dataset.ESPnetEZDataset(eval_hf, data_info=test_data_info)

print("Train dataset first item:", train_dataset[0])
print("Validation dataset first item:", valid_dataset[0])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_fn(args):
    model = pretrained_model.s2t_model
    model.train()
    print(f'Trainable parameters: {count_parameters(model)}')
    return model

EXP_DIR = f"./exp/finetune"
STATS_DIR = f"./exp/stats_finetune"
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

finetune_config = ez.config.update_finetune_config(
    's2t',
    pretrain_config,
    f"/ocean/projects/cis250187p/dgarg2/Speech-Project/owsm_finetuning/config/finetune.yaml"
)

trainer = ez.Trainer(
    task='s2t',
    train_config=finetune_config,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    build_model_fn=build_model_fn,
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1
)

trainer.collect_stats()

trainer.train()

# Generate transcripts for all test/eval data
print("\n" + "="*80)
print("Generating transcripts for test/eval data...")
print("="*80)

# Load finetuned model
d = torch.load("./exp/finetune/5epoch.pth")
pretrained_model.s2t_model.load_state_dict(d)
pretrained_model.s2t_model.eval()

print(f"Model device: {pretrained_model.device}")
print(f"Total test samples: {len(eval_hf)}")

# Generate transcripts for all eval samples and save to CSV
predictions_data = []

for i in range(len(eval_hf)):
    id, sample = test_dataset.__getitem__(i)
    try:
        pred = pretrained_model(sample['speech'])
        predicted_text = pred[0][3]
        reference_text = sample['text_raw']
        file_id = eval_hf[i]['file_id']
        
        predictions_data.append({
            "File-id": file_id,
            "PREDICTED": predicted_text,
            "REFERENCE": reference_text
        })
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(eval_hf)} samples...")
            
    except Exception as e:
        print(f"Error processing sample {i} (file_id: {eval_hf[i]['file_id']}): {e}")
        # Still add to CSV with error message
        predictions_data.append({
            "File-id": eval_hf[i]['file_id'],
            "PREDICTED": f"ERROR: {str(e)}",
            "REFERENCE": eval_hf[i]['transcription'] if i < len(eval_hf) else "N/A"
        })
        continue

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions_data)
output_csv = "./test_predictions.csv"
predictions_df.to_csv(output_csv, index=False)

print("\n" + "="*80)
print(f"Finished generating transcripts. Saved {len(predictions_data)} predictions to {output_csv}")
print("="*80)
