# ########################################################    
# # R: This code take the embeddings and attention mask and decode it to get the answer using vt5 model (IN THIS CASE WE USE T5-BASE)
# ########################################################


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import yaml


def decode_using_saved_embeddings():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load configuration
    config_path = r"/home/rriccio/Desktop/GraphDoc/config/models/vt5.yml"  # Adjust the path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load VT5 decoder
    model = T5ForConditionalGeneration.from_pretrained(config['model_weights'])
    model = model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(config['model_weights'])

    # EMBEDDIN FROM VT5:

    # Load precomputed embeddings and attention mask
    # encoder_output_path = r"/home/rriccio/Desktop/GraphDoc/samples/input_embeds.pt"
    # attention_mask_path = r"/home/rriccio/Desktop/GraphDoc/samples/attention_mask.pt"
    # input_embeds = torch.load(encoder_output_path, map_location=device)
    # attention_mask = torch.load(attention_mask_path, map_location=device)

    #EMBEDDING FROM GRAPHDOC:
    # Load precomputed embeddings and attention mask
    embedding_file_path = r"/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4_embeddings.pt"  # Adjust the path
    saved_data = torch.load(embedding_file_path, map_location=device)

    # Extract relevant data
    input_embeds = saved_data['last_hidden_state']  # Use 'last_hidden_state' for embeddings
    attention_mask = saved_data['attention_mask']  # Use the saved attention mask


    # **Verification Step**: Print shapes to confirm
    print(f"Loaded input_embeds shape: {input_embeds.shape}")  # Expected: [batch_size, seq_length, embed_dim]
    print(f"Loaded attention_mask shape: {attention_mask.shape}")  # Expected: [batch_size, seq_length]

    try:
        # Create decoder_input_ids (start with PAD token as in T5)
        batch_size = input_embeds.size(0)
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * tokenizer.pad_token_id

        # Generate the answer
        output = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=32,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
            decoder_input_ids=decoder_input_ids,
        )
        
        # Decode the generated answer
        pred_answer = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
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
    decode_using_saved_embeddings()
