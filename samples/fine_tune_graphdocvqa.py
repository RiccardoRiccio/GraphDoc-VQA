

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import os
import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
import numpy as np
from logger import Logger
from metrics import Evaluator
from pathlib import Path
from tqdm import tqdm

# Utility to load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Dataset for precomputed embeddings
class PrecomputedDocVQADataset(Dataset):
    def __init__(self, pt_dir, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.pt_files = list(Path(pt_dir).glob("**/*.pt"))
        # print(f"Dataset initialized with {len(self.pt_files)} files in {pt_dir}")  # Confirm initialization
    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        pt_file = self.pt_files[idx]
        data = torch.load(pt_file)

        # Precomputed embeddings
        last_hidden_state = data["last_hidden_state"].squeeze(0)
        attention_mask = data["attention_mask"].squeeze(0)

        # Assuming 'answers' contains a list of all valid answers
        all_answers = data.get("answers", [])
        target_answer = all_answers[0] if all_answers else "Unknown Answer"
        question = data.get("question", "Unknown Question")
        image_path = data.get("image_path", "Unknown Image")

        # Tokenize the first answer for training purposes
        labels = self.tokenizer(
            target_answer,
            max_length=self.config["max_source_length"],
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "inputs_embeds": last_hidden_state,
            "attention_mask": attention_mask,
            "labels": labels,
            "question": question,
            "image_path": image_path,
            "all_answers": all_answers  # Include all valid answers
        }




# VT5 Model wrapper
class VT5Model(torch.nn.Module):
    def __init__(self, vt5_model_path):
        super(VT5Model, self).__init__()
        self.vt5_decoder = T5ForConditionalGeneration.from_pretrained(vt5_model_path)

    def forward(self, inputs_embeds, attention_mask, labels=None, return_pred_answer=False):
        # print(f"inputs_embeds shape: {inputs_embeds.shape}")  # Should be [batch_size, seq_length, 768]
        # print(f"attention_mask shape: {attention_mask.shape}")  # Should be [batch_size, seq_length]
        # print(f"labels shape: {labels.shape if labels is not None else None}")  # Should be [batch_size, label_len]

        if labels is not None:
            decoder_input_ids = labels[:, :-1]  # Shift for decoder
            shifted_labels = labels[:, 1:]
        else:
            decoder_input_ids = None
            shifted_labels = None

        outputs = self.vt5_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )


        if return_pred_answer:
            pred_ids = self.vt5_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=4,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
                early_stopping=True,
            )
            return outputs, pred_ids
        return outputs


# Collate function for dataloader
def collate_fn(batch):
    max_seq_len = max(b["inputs_embeds"].size(0) for b in batch)
    max_label_len = max(b["labels"].size(0) for b in batch)

    inputs_embeds = torch.stack([
        torch.nn.functional.pad(
            b["inputs_embeds"],
            (0, 0, 0, max_seq_len - b["inputs_embeds"].size(0)),
            value=0.0
        )
        for b in batch
    ])

    attention_mask = torch.stack([
        torch.nn.functional.pad(
            b["attention_mask"],
            (0, max_seq_len - b["attention_mask"].size(0)),
            value=0
        )
        for b in batch
    ])

    labels = torch.stack([
        torch.nn.functional.pad(
            b["labels"],
            (0, max_label_len - b["labels"].size(0)),
            value=-100
        )
        for b in batch
    ])

    questions = [b["question"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    all_answers = [b["all_answers"] for b in batch]  # Collect all valid answers

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
        "questions": questions,
        "image_paths": image_paths,
        "all_answers": all_answers  # Include all valid answers in the batch
    }







# Training epoch
def train_epoch(data_loader, model, optimizer, scheduler, evaluator, logger, device):
    model.train()
    # Add tqdm to track the progress of batches
    with tqdm(data_loader, desc="Training", unit="batch") as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Move only tensor values to the device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            outputs = model(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log_train_metrics(loss=loss.item(), lr=current_lr, step=batch_idx)

            # Update tqdm with the current loss
            pbar.set_postfix(loss=loss.item(), lr=current_lr)


# Evaluation function
def evaluate(data_loader, model, tokenizer, evaluator, device):
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            batch_size = batch["inputs_embeds"].size(0)
            decoder_input_ids = torch.ones(
                (batch_size, 1), dtype=torch.long, device=device
            ) * tokenizer.pad_token_id

            pred_ids = model.vt5_decoder.generate(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                max_length=32,
                num_beams=4,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
                early_stopping=True,
            )
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            refs = batch["all_answers"]  # Use all valid answers for evaluation

            # for image_path, question, pred, ref in zip(batch["image_paths"], batch["questions"], preds, refs):
                # print(f"Image: {image_path}")
                # print(f"Question: {question}")
                # print(f"Prediction: {pred}")
                # print(f"References: {ref}\n")

            all_preds.extend(preds)
            all_refs.extend(refs)

    metrics = evaluator.get_metrics(all_refs, all_preds)

    mean_accuracy = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
    mean_anls = np.mean(metrics['anls']) if metrics['anls'] else 0

    return mean_accuracy, mean_anls





# Main function
def main():
    # Load configuration
    config = load_config("config/models/vt5.yml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model_path = config["model_weights"]
    model = VT5Model(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Datasets and Dataloaders
    train_dataset = PrecomputedDocVQADataset(
        pt_dir=config["dataset_paths"]["train_embeddings_dir"],
        tokenizer=tokenizer,
        config=config
    )
    val_dataset = PrecomputedDocVQADataset(
        pt_dir=config["dataset_paths"]["val_embeddings_dir"],
        tokenizer=tokenizer,
        config=config
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    # Logger and Evaluator
    logger = Logger(config)
    evaluator = Evaluator()

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=float(config["training_parameters"]["lr"]))
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=config["training_parameters"]["warmup_iterations"],
        num_training_steps=config["training_parameters"]["train_epochs"] * len(train_loader)
    )

    best_accuracy = 0.0
    for epoch in range(config["training_parameters"]["train_epochs"]):
        print(f"Starting epoch {epoch}...")
        train_epoch(train_loader, model, optimizer, scheduler, evaluator, logger, device)
        accuracy, anls = evaluate(val_loader, model, tokenizer, evaluator, device)
        
        # Compute mean of metrics
        mean_accuracy = np.mean(accuracy) if isinstance(accuracy, list) else accuracy
        mean_anls = np.mean(anls) if isinstance(anls, list) else anls
        
         # Epoch summary
        print(f"Epoch {epoch} - Mean Accuracy: {mean_accuracy:.4f}, Mean ANLS: {mean_anls:.4f}")



        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            model.vt5_decoder.save_pretrained(config["save_dir"])
            tokenizer.save_pretrained(config["save_dir"])


if __name__ == "__main__":
    main()