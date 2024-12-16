# CHECK NUMBER OF PARAMETERES OF GRAPHDOC AND VT5
import sys
sys.path.append('.')
import torch
from transformers import T5ForConditionalGeneration
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode  # Import your GraphDoc encoder model

# Load pretrained encoder (GraphDoc)
graphdoc_path = "pretrained_model/graphdoc"  # Adjust this path based on your folder structure
graphdoc = GraphDocForEncode.from_pretrained(graphdoc_path)

# Load pretrained decoder (VT5)
vt5_path = "rubentito/vt5-base-spdocvqa"  # Adjust this path based on your decoder weights
vt5_model = T5ForConditionalGeneration.from_pretrained(vt5_path)

# Function to print parameter details
def print_model_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n{model_name} Model Parameters:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {frozen_params:,}")
    print(f"  List of Layers and Parameter Count:")
    for name, param in model.named_parameters():
        print(f"    {name}: {param.numel():,} {'(Trainable)' if param.requires_grad else '(Frozen)'}")

# Print GraphDoc Encoder parameters
print_model_parameters(graphdoc, "GraphDoc Encoder")

# Print VT5 Decoder parameters
print_model_parameters(vt5_model, "VT5 Decoder")
