from transformers import T5ForConditionalGeneration

# Load the T5 decoder (bypassing the encoder)
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Decoder only: Ensure encoder is skipped
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

decoder_params = sum(p.numel() for n, p in model.named_parameters() if "decoder" in n)
print(f"Decoder parameters: {decoder_params:,}")


print(f"Total parameters in full model: {total_params:,}")  # Should be 222M
print(f"Trainable parameters (decoder only): {trainable_params:,}")  # Should match decoder params
