
import torch

def compare_specific_tensors(single_file, batch_file, key):
    # Load the .pt files
    single_data = torch.load(single_file)
    batch_data = torch.load(batch_file)

    print(f"=== Detailed Check for Key: {key} ===")
    
    single_value = single_data[key]
    batch_value = batch_data[key]

    if isinstance(single_value, torch.Tensor):
        # Ensure both tensors are on the same device
        single_value = single_value.cpu()
        batch_value = batch_value.cpu()

        print(f"Single Tensor Shape: {single_value.shape}")
        print(f"Batch Tensor Shape: {batch_value.shape}")
        
        # Check if shapes match
        if single_value.shape == batch_value.shape:
            print("Shapes Match")
        else:
            print("Shapes Mismatch!")

        # For attention masks, print all values
        if key == "attention_mask":
            print("\nSingle Attention Mask Values:")
            print(single_value)
            
            print("\nBatch Attention Mask Values:")
            print(batch_value)

        # For tensors with shape [1, NODES, 768], print specific values
        if single_value.ndim == 3 and single_value.shape[-1] == 768:
            nodes = single_value.shape[1]
            print("\nSingle Tensor Values (First and Last Node, First 10 and Last 10 Features):")
            print("First Node, First 10 Features:")
            print(single_value[0, 0, :10])
            print("First Node, Last 10 Features:")
            print(single_value[0, 0, -10:])
            
            print("Last Node, First 10 Features:")
            print(single_value[0, -1, :10])
            print("Last Node, Last 10 Features:")
            print(single_value[0, -1, -10:])

            print("\nBatch Tensor Values (First and Last Node, First 10 and Last 10 Features):")
            print("First Node, First 10 Features:")
            print(batch_value[0, 0, :10])
            print("First Node, Last 10 Features:")
            print(batch_value[0, 0, -10:])
            
            print("Last Node, First 10 Features:")
            print(batch_value[0, -1, :10])
            print("Last Node, Last 10 Features:")
            print(batch_value[0, -1, -10:])
    else:
        print("The selected key does not contain a tensor!")
        print(f"Single Value: {single_value}")
        print(f"Batch Value: {batch_value}")

def main():
    # File paths
    single_file = "/data2/users/rriccio/spdocvqa_embeddings_sample_original/ffng0227_13.pt"
    batch_file = "/data2/users/rriccio/spdocvqa_embeddings_sample_original/ffng0227_13.pt"

    # Check specific keys
    for key in ['last_hidden_state', 'attention_mask']:
        compare_specific_tensors(single_file, batch_file, key)

if __name__ == "__main__":
    main()