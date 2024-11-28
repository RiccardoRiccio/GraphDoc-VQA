import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutlmft.models.vqa.vt5 import ProxyVT5
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import yaml

def decode_using_saved_encoder():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load configuration from YAML
    config_path = r"/home/rriccio/Desktop/GraphDoc/config/models/vt5.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Extract relevant config parts and modify for base T5
    model_config = {
        'batch_size': 1,  # Set to 1 for inference
        'model_weights': 't5-base',  # Use base T5 model instead
        'visual_module': config['visual_module'],
        'max_source_length': config['max_source_length']
    }
    
    try:
        # Initialize model
        model = ProxyVT5(model_config)
        # Move model components to device
        model.model = model.model.to(device)
        model.spatial_embedding = model.spatial_embedding.to(device)
        model.visual_embedding = model.visual_embedding.to(device)
        
        # Define paths to saved encoder outputs
        encoder_output_path = r"/home/rriccio/Desktop/GraphDoc/samples/input_embeds.pt"
        attention_mask_path = r"/home/rriccio/Desktop/GraphDoc/samples/attention_mask.pt"
        
        # Load saved encoder outputs
        input_embeds = torch.load(encoder_output_path, map_location=device)
        attention_mask = torch.load(attention_mask_path, map_location=device)
        
        # **Verification Step**: Print shapes to confirm
        print(f"Loaded input_embeds shape: {input_embeds.shape}")  # Expected: [1, sequence_length, embed_dim]
        print(f"Loaded attention_mask shape: {attention_mask.shape}")  # Expected: [1, sequence_length]

         # Create decoder_input_ids before generate call
        batch_size = input_embeds.size(0)
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * model.tokenizer.pad_token_id

        
        # Generate answer using input_embeds and attention_mask
        output = model.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=32,
            min_length=1,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            decoder_input_ids=decoder_input_ids,
            bos_token_id=model.tokenizer.pad_token_id,  # which is 0
            decoder_start_token_id=model.tokenizer.pad_token_id,  # which is 0
        )
        
        # Decode the generated answer
        pred_answer = model.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        
        print(f"Answer: {pred_answer}")
        
        # Print confidence if available
        if hasattr(output, 'sequences_scores') and output.sequences_scores is not None:
            confidence = torch.exp(output.sequences_scores[0]).item()
            print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

if __name__ == "__main__":
    decode_using_saved_encoder()
