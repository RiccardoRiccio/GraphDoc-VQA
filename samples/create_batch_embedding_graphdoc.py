import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from runner.graphdoc.encode_batch_document import get_document_embedding
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging

# Suppress warnings from the transformers library
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
torch.cuda.empty_cache()
def save_embeddings_for_split(data_split, qas_file, image_dir, ocr_dir, save_dir, batch_size=8):
    """
    Precompute and save embeddings for a specific dataset split using batches.
    
    Args:
        data_split (str): One of 'train', 'val', or 'test'.
        qas_file (str): Path to the JSON file containing the dataset split.
        image_dir (str): Directory containing images.
        ocr_dir (str): Directory containing OCR JSON files.
        save_dir (str): Directory to save the embeddings.
        batch_size (int): Number of samples to process in a single batch.
    """
    # Load dataset
    with open(qas_file, 'r') as f:
        dataset = json.load(f)

    samples = dataset["data"]
    split_save_dir = os.path.join(save_dir, data_split)
    os.makedirs(split_save_dir, exist_ok=True)

    batch_data = []  # Temporary storage for batch

    for sample in tqdm(samples, desc=f"Processing {data_split} split"):
        question = sample["question"]
        answers = sample.get("answers", [])
        question_id = sample["questionId"]
        image_name = Path(sample["image"]).name
        image_path = os.path.join(image_dir, image_name)
        ocr_path = os.path.join(ocr_dir, Path(image_name).stem + ".json")

        batch_data.append((question, image_path, ocr_path, question_id, answers, image_name))

        # Process batch when batch size is reached
        if len(batch_data) == batch_size:
            process_batch(batch_data, split_save_dir)
            batch_data = []  # Clear batch

    # Process remaining samples in the last batch
    if batch_data:
        process_batch(batch_data, split_save_dir)


def process_batch(batch_data, split_save_dir):
    """
    Process a batch of samples and save their embeddings individually.
    
    Args:
        batch_data (list): List of sample data in the batch.
        split_save_dir (str): Directory to save the embeddings.
    """

    torch.cuda.empty_cache()

    questions, image_paths, ocr_paths, question_ids, answers_list, image_names = zip(*batch_data)

    # print(f"Processing batch: {len(batch_data)} samples")
    # print(f"First image path in batch: {image_paths[0]}")

    try:
        # Get batch embeddings
        last_hidden_states, pooler_outputs, attention_masks = get_document_embedding(
            questions=list(questions), 
            image_paths=list(image_paths), 
            ocr_paths=list(ocr_paths)
        )

        # torch.cuda.empty_cache()

        # Save embeddings for each sample
        for idx, (hidden_state, pooler_output, attention_mask, question, answers, image_name, question_id) in enumerate(
                zip(last_hidden_states, pooler_outputs, attention_masks, questions, answers_list, image_names, question_ids)):

            # Ensure tensors retain the batch dimension
            hidden_state = hidden_state.unsqueeze(0) if hidden_state.ndim == 2 else hidden_state
            pooler_output = pooler_output.unsqueeze(0) if pooler_output.ndim == 1 else pooler_output
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
            
            save_path = os.path.join(split_save_dir, f"{Path(image_name).stem}_q{question_id}.pt")
            torch.save({
                "last_hidden_state": hidden_state.cpu(),  # Ensure tensor is moved to CPU before saving
                "pooler_output": pooler_output.cpu(),
                "attention_mask": attention_mask.cpu(),
                "question": question,
                "answers": answers,
                "image_path": image_paths[idx],
                "ocr_path": ocr_paths[idx]
            }, save_path)

    except Exception as e:
        print(f"Error processing batch: {e}")
        print(f"Batch data details: {batch_data}")
        raise  # Re-raise error for debugging


def main():
    # Directories and file paths
    image_dir = "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_images"
    ocr_dir = "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_ocr"
    save_dir = "/home/rriccio/Desktop/GraphDoc/spdocvqa_embeddings_sample"

    # JSON files for each split
    splits = {
        # "train": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/train_v1.0_withQT.json",
        # "val": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/val_v1.0_withQT.json",
        # "test": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/test_v1.0.json"

        "train": "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_qas/train_v1.0_withQT.json",
        "val": "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_qas/val_v1.0_withQT.json",
        "test": "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_qas/test_v1.0.json"
    }

    # Save embeddings for each split
    for split, qas_file in splits.items():
        # print(f"Processing {split} split...")
        save_embeddings_for_split(split, qas_file, image_dir, ocr_dir, save_dir)


if __name__ == "__main__":
    main()
