import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys

# Load precomputed .pt file and VT5 model
def load_model_and_predict(pt_file_path, model_path):
    # Load .pt file
    data = torch.load(pt_file_path, map_location="cpu")

    # Extract relevant data from the .pt file
    last_hidden_state = data.get("last_hidden_state")
    attention_mask = data.get("attention_mask")
    question = data.get("question")
    answers = data.get("answers")

    if last_hidden_state is None or attention_mask is None:
        raise ValueError("Invalid .pt file. Missing embeddings or attention mask.")

    print(f"Question: {question}")
    print(f"Ground Truth Answers: {answers}")

    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Convert inputs to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate prediction
    model.eval()
    with torch.no_grad():
        pred_ids = model.generate(
            inputs_embeds=torch.tensor(last_hidden_state).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
            max_length=32,
            num_beams=4,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            early_stopping=True,
        )

    # Decode prediction
    pred_answer = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    print(f"Predicted Answer: {pred_answer}")

# Main function
def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_answer_debug.py <path_to_pt_file> <path_to_model>")
        sys.exit(1)

    pt_file_path = sys.argv[1]
    model_path = sys.argv[2]

    load_model_and_predict(pt_file_path, model_path)

if __name__ == "__main__":
    main()
