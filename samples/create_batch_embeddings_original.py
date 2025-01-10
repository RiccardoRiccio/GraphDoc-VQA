####################
# SAVE IN A REPOSITORY THE EMBEDDINGS USING THE ORIGINAL GRAPHDOC
#######################


import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from runner.graphdoc.encode_original_batch_document import get_original_document_embedding
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import gc

# Suppress warnings from the transformers library
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
torch.cuda.empty_cache()

def save_image_embeddings(image_dir, ocr_dir, save_dir, batch_size=8):
    """
    Precompute and save embeddings for images and corresponding OCR files in batches.

    Args:
        image_dir (str): Directory containing images.
        ocr_dir (str): Directory containing OCR JSON files.
        save_dir (str): Directory to save the embeddings.
        batch_size (int): Number of samples to process in a single batch.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    batch_data = []  # Temporary storage for batch

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        ocr_path = os.path.join(ocr_dir, Path(image_file).stem + ".json")

        if not os.path.exists(ocr_path):
            print(f"OCR file missing for {image_file}, skipping.")
            continue

        # Append data for processing (no questions involved)
        batch_data.append((image_path, ocr_path, image_file))

        # Process batch when batch size is reached
        if len(batch_data) == batch_size:
            process_batch(batch_data, save_dir)
            batch_data = []  # Clear batch

    # Process remaining samples in the last batch
    if batch_data:
        process_batch(batch_data, save_dir)


def process_batch(batch_data, save_dir):
    """
    Process a batch of image and OCR pairs to generate and save embeddings.

    Args:
        batch_data (list): List of tuples (image_path, ocr_path, image_file).
        save_dir (str): Directory to save the embeddings.
    """
    image_paths, ocr_paths, image_files = zip(*batch_data)

    try:
        with torch.no_grad():  # <<< Added here
            # Generate embeddings for the batch
            last_hidden_states, pooler_outputs, attention_masks = get_original_document_embedding(
                image_paths=list(image_paths),
                ocr_paths=list(ocr_paths)
            )

        # Save embeddings for each sample in the batch
        for idx, (hidden_state, pooler_output, attention_mask, image_file) in enumerate(
            zip(last_hidden_states, pooler_outputs, attention_masks, image_files)
        ):
            # Ensure tensors retain the batch dimension
            hidden_state = hidden_state.unsqueeze(0) if hidden_state.ndim == 2 else hidden_state
            pooler_output = pooler_output.unsqueeze(0) if pooler_output.ndim == 1 else pooler_output
            attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

            save_path = os.path.join(save_dir, f"{Path(image_file).stem}.pt")
            torch.save({
                "last_hidden_state": hidden_state.cpu(),
                "pooler_output": pooler_output.cpu(),
                "attention_mask": attention_mask.cpu(),
                "image_path": image_paths[idx],
                "ocr_path": ocr_paths[idx]
            }, save_path)
        
        # Clear GPU memory
        del last_hidden_states, pooler_outputs, attention_masks
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing batch: {e}")
        print(f"Batch data details: {batch_data}")
        raise


def main():
    # Directories and file paths
    # image_dir = "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_images"
    image_dir = "/data2/users/rriccio/spdocvqa_images"
    ocr_dir = "/data2/users/rriccio/spdocvqa_ocr"
    save_dir = "/data2/users/rriccio/spdocvqa_embeddings_sample_original"


    # Save embeddings for all images in batches
    save_image_embeddings(image_dir, ocr_dir, save_dir, batch_size=128)

if __name__ == "__main__":
    main()
