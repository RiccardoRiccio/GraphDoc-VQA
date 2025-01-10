import torch
from transformers import T5ForConditionalGeneration

def check_t5_parameter_status(model_name="t5-base", freeze_encoder=True):
    # Load T5 model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    if freeze_encoder:
        # Freeze the encoder and shared embeddings
        for p in model.encoder.parameters():
            p.requires_grad = False
        # for p in model.shared.parameters():  # Embedding layer
        #     p.requires_grad = False

        # Optionally, freeze lm_head (final linear layer) if needed
        # for p in model.lm_head.parameters():
            # p.requires_grad = True
            # print("lm head p: ", p)

    # Print the status of each parameter
    print("\n====== PARAMETER FREEZING STATUS ======\n")
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        print(f"{name.ljust(70)} -> {status}")
    
    # Count frozen and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("\n====== SUMMARY ======\n")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")
    print(f"Frozen Parameters: {frozen_params:,} ({(frozen_params / total_params) * 100:.2f}%)")

if __name__ == "__main__":
    check_t5_parameter_status("rubentito/vt5-base-spdocvqa", freeze_encoder=True)
