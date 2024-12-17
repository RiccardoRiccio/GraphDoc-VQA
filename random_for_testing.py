import torch

def inspect_pt_file(file_path):
    # Load the .pt file
    data = torch.load(file_path)
    
    print(f"Inspecting contents of: {file_path}\n")
    
    # Iterate through keys and print details
    for key, value in data.items():
        print(f"Key: {key}")
        
        if isinstance(value, torch.Tensor):
            print(f"Type: Tensor")
            print(f"Shape: {value.shape}")
            print(f"First 10 elements (flattened): {value.view(-1)[:10]}\n")
        elif isinstance(value, str):
            print(f"Type: String")
            print(f"Value: {value}\n")
        else:
            print(f"Type: {type(value)}")
            print(f"Value: {value}\n")

def main():
    file_path = "/home/rriccio/Desktop/GraphDoc/spdocvqa_embeddings_sample_original/ffng0227_13.pt"
    inspect_pt_file(file_path)

if __name__ == "__main__":
    main()
