import os
import shutil
import json

print("hi")

# Paths
images_src_dir = "sp-docvqa/spdocvqa_images"  # Full dataset images directory
images_dest_dir = "sp-docvqa_sample/spdocvqa_images_sample"  # Sample dataset directory
json_files = {
    "train": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/train_v1.0_withQT.json",
    "val": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/val_v1.0_withQT.json",
    "test": "/home/rriccio/Desktop/GraphDoc/sp-docvqa_sample/spdocvqa_qas_sample/test_v1.0.json"
}

# Ensure the base destination directory exists
os.makedirs(images_dest_dir, exist_ok=True)

# Create subfolders for train, val, and test
for split in json_files.keys():
    split_dir = os.path.join(images_dest_dir, split)
    os.makedirs(split_dir, exist_ok=True)  # Create subfolder if it doesn't exist

# Function to copy images
def copy_images(json_path, split_name):
    print(f"Processing {split_name} dataset...")
    with open(json_path, "r") as f:
        data = json.load(f)
    for entry in data["data"]:
        # Extract the image name (ignoring any subfolders)
        image_name = os.path.basename(entry["image"])  # Removes "documents/" if it exists
        src_path = os.path.join(images_src_dir, image_name)  # Directly look in spdocvqa_images
        dest_path = os.path.join(images_dest_dir, split_name, image_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied: {src_path} -> {dest_path}")
        else:
            print(f"Image not found: {src_path}")

# Copy images for each split
for split, json_path in json_files.items():
    if os.path.exists(json_path):
        copy_images(json_path, split)
    else:
        print(f"JSON file not found: {json_path}")
