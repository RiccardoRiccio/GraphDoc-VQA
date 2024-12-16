from datasets import load_dataset

# Specify the directory with enough space
cache_directory = "/home/rriccio/dataset_cache"

# Load the dataset into the specified directory
dataset = load_dataset("aharley/rvl_cdip", cache_dir=cache_directory)

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# Example: Processing images and labels
for sample in train_dataset:
    image = sample['image']
    label = sample['label']
    # Process the image and label as needed
