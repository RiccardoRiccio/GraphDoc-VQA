import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from runner.graphdoc.encode_document import get_document_embedding


def save_embeddings_for_split(data_split, qas_file, image_dir, ocr_dir, save_dir):
    """
    Precompute and save embeddings for a specific dataset split.
    
    Args:
        data_split (str): One of 'train', 'val', or 'test'.
        qas_file (str): Path to the JSON file containing the dataset split.
        image_dir (str): Directory containing images.
        ocr_dir (str): Directory containing OCR JSON files.
        save_dir (str): Directory to save the embeddings.
    """
    # Load dataset
    with open(qas_file, 'r') as f:
        dataset = json.load(f)

    samples = dataset["data"]
    split_save_dir = os.path.join(save_dir, data_split)
    os.makedirs(split_save_dir, exist_ok=True)

    for sample in tqdm(samples, desc=f"Processing {data_split} split"):
        question = sample["question"]
        answers = sample.get("answers", [])
        question_id = sample["questionId"]
        image_name = Path(sample["image"]).name  # Extract only the filename (e.g., "mxxj0037_1.png")
        image_path = os.path.join(image_dir, image_name)  # Combine with the base image directory
        ocr_path = os.path.join(ocr_dir, Path(image_name).stem + ".json")

        # Compute embeddings using get_document_embedding
        try:
            last_hidden_state, pooler_output, attention_mask = get_document_embedding(question, image_path, ocr_path)

            # Save embeddings and metadata
            save_path = os.path.join(split_save_dir, f"{Path(image_name).stem}_q{question_id}.pt")
            torch.save({
                "last_hidden_state": last_hidden_state,
                "pooler_output": pooler_output,
                "attention_mask": attention_mask,
                "question": question,
                "answers": answers,
                "image_path": image_path,
                "ocr_path": ocr_path
            }, save_path)

            # print(f"Saved embeddings for question {question_id} to {save_path}")
        except Exception as e:
            print(f"Error processing {image_name} for question {question_id}: {e}")


def main():
    # Directories and file paths
    image_dir = "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_images"
    ocr_dir = "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_ocr"
    save_dir = "/home/rriccio/Desktop/GraphDoc/spdocvqa_embeddings_single"

    # JSON files for each split
    splits = {
        "train": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/train_v1.0_withQT.json",
        "val": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/val_v1.0_withQT.json",
        "test": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/test_v1.0.json"
    }

    # Save embeddings for each split
    for split, qas_file in splits.items():
        print(f"Processing {split} split...")
        save_embeddings_for_split(split, qas_file, image_dir, ocr_dir, save_dir)


if __name__ == "__main__":
    main()
