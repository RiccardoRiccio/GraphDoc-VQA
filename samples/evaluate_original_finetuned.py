##########################
##### EVALUATE AND PRINT GT AND PRED OF THE FINE TUNE MODEL MADE ON "FINE_TUNE_GRAPHDOCVQA.PY"
##########################


import torch
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
from pathlib import Path
from tqdm import tqdm
import random
from fine_tune_graphdocvqa_version2 import DocVQAQuestionDrivenDataset, docvqa_collate_fn, load_config
from metrics import Evaluator


def load_finetuned_model(config):
    print("[Info] Loading fine-tuned model...")
    model = T5ForConditionalGeneration.from_pretrained(config["save_dir"])
    tokenizer = T5Tokenizer.from_pretrained(config["save_dir"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def evaluate_and_print_results(data_loader, model, tokenizer, evaluator, device, print_limit=20):
    print("[Info] Starting evaluation on validation data...")

    all_preds = []
    all_refs = []
    printed_count = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            questions = batch["questions"]
            doc_embeds_list = batch["doc_embeds"]
            doc_masks_list = batch["doc_masks"]
            gt_answers = batch["all_answers"]

            B = len(questions)
            max_doc_len = max(e.size(0) for e in doc_embeds_list)
            d_model = doc_embeds_list[0].size(-1)

            padded_doc_embeds = torch.zeros(B, max_doc_len, d_model).to(device)
            padded_doc_mask = torch.zeros(B, max_doc_len, dtype=torch.long).to(device)

            for i in range(B):
                seq_len = doc_embeds_list[i].size(0)
                padded_doc_embeds[i, :seq_len, :] = doc_embeds_list[i].to(device)
                padded_doc_mask[i, :seq_len] = doc_masks_list[i].to(device)

            tokens = tokenizer(
                ["question: " + q for q in questions],
                max_length=512,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            q_embeds = model.shared(tokens.input_ids)
            input_embeds = torch.cat([q_embeds, padded_doc_embeds], dim=1)
            attention_mask = torch.cat([tokens.attention_mask, padded_doc_mask], dim=1)

            outputs = model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=4,
                decoder_input_ids=torch.ones((B, 1), dtype=torch.long, device=device) * tokenizer.pad_token_id
            )


            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Print predictions and ground truth (First 20 only)
            for i, pred in enumerate(predictions):
                if printed_count < print_limit:
                    print("\nQuestion:", questions[i])
                    print("Prediction:", pred)
                    print("Ground Truth:", gt_answers[i])
                    printed_count += 1

                all_preds.append(pred)
                all_refs.append(gt_answers[i])

    print("\n[Info] Evaluating predictions...")
    metric = evaluator.get_metrics(all_refs, all_preds)
    accuracy = float(torch.tensor(metric["accuracy"]).mean())
    anls = float(torch.tensor(metric["anls"]).mean())

    print(f"\n[Final Results] Accuracy: {accuracy:.4f}, ANLS: {anls:.4f}")


def main():
    config = load_config("config/models/vt5.yml")
    model, tokenizer, device = load_finetuned_model(config)

    val_dataset = DocVQAQuestionDrivenDataset(
        embeddings_dir=config["dataset_paths"]["val_embeddings_dir"],
        qa_json_path=config["dataset_paths"]["val_qas_path"],
        tokenizer=tokenizer,
        config=config
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=docvqa_collate_fn
    )

    evaluator = Evaluator()

    # Print only the first 20 results
    evaluate_and_print_results(val_loader, model, tokenizer, evaluator, device, print_limit=20)


if __name__ == "__main__":
    main()
