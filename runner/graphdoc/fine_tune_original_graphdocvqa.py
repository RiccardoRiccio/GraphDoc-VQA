import os
import sys
import yaml
import json
import torch
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# import models._model_utils as model_utils
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from torch.optim import AdamW
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from logger import Logger
from metrics import Evaluator

#######################
# PART TO ,MODIFY:  1) verify the tokenizer for q and a 2) check t5generator if pasing correct parameters

######################
# Utility Functions
######################
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_qas(json_path):
    """Load the QAs from the given JSON. 
    Returns a dictionary mapping image base name (without extension) -> list of QAs.
    Each QA is a dict with keys: question, answers.
    Example image key: 'xnbl0037_1' corresponds to 'documents/xnbl0037_1.png'
    We will strip 'documents/' and '.png' to map to embedding file 'xnbl0037_1.pt'."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    qas_map = {}
    for item in data["data"]:
        image_path = item["image"]  # e.g. documents/xnbl0037_1.png
        base_name = Path(image_path).stem  # e.g. xnbl0037_1
        question = item["question"]
        answers = item["answers"]  # list of possible answers
        if base_name not in qas_map:
            qas_map[base_name] = []
        qas_map[base_name].append({
            "question": question,
            "answers": answers
        })
    return qas_map


######################
# Dataset
######################
class PrecomputedDocVQADataset(Dataset):
    def __init__(self, embeddings_dir, qas_json_path, tokenizer, config, model, missing_log_path=None):
        """
        embeddings_dir: Directory with all .pt files for document images.
        qas_json_path: Path to the JSON file with QAs.
        tokenizer: T5 tokenizer.
        model: A T5 model to access embeddings layer for question tokens.
        config: Configuration dict.
        missing_log_path: File path to save the missing image-question pairs.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.embeddings_dir = Path(embeddings_dir)
        self.missing_log_path = missing_log_path
        
        # Load QAs
        self.qas_map = load_qas(qas_json_path)
        
        # Create a list of (pt_file_path, question, answers) tuples
        self.samples = []
        self.missing_entries = []  # To log missing entries
        for base_name, qas_list in self.qas_map.items():
            pt_path = self.embeddings_dir / f"{base_name}.pt"  # Look for embeddings in the specified folder
            if pt_path.exists():
                for qa in qas_list:
                    self.samples.append((pt_path, qa["question"], qa["answers"]))
            else:
                # Log missing image-question pairs
                for qa in qas_list:
                    self.missing_entries.append({
                        "image": base_name,
                        "question_id": qa.get("question_id", "unknown"),
                        "question": qa["question"]
                    })

        # Save missing entries to file if specified
        if self.missing_log_path and self.missing_entries:
            with open(self.missing_log_path, "w") as f:
                json.dump(self.missing_entries, f, indent=4)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, question, answers = self.samples[idx]
        data = torch.load(pt_path)

        # Document embeddings
        doc_hidden_state = data["last_hidden_state"].squeeze(0)  # Convert to [seq_len, hidden_size]
        doc_attention_mask = data["attention_mask"].squeeze(0)  # Convert to [seq_len]


        # Tokenize the question
        question_enc = self.tokenizer(
            question,
            max_length=self.config["max_source_length"],
            truncation=True,
            return_tensors="pt"
        )
        
        question_input_ids = question_enc.input_ids[0]     # [q_len]
        question_attention_mask = question_enc.attention_mask[0]  # [q_len]

        # Convert question_input_ids to embeddings using T5 encoder embedding layer
        question_embeddings = self.model.vt5_decoder.get_input_embeddings()(question_input_ids.unsqueeze(0))
        question_embeddings = question_embeddings.squeeze(0)  # [q_len, hidden_size]

        # Concatenate question and doc embeddings along seq dimension
        combined_inputs_embeds = torch.cat([question_embeddings, doc_hidden_state], dim=0)
        combined_attention_mask = torch.cat([question_attention_mask, doc_attention_mask], dim=0)

        # Choose the first answer as target (or a random answer)
        target_answer = np.random.choice(answers) if len(answers) > 0 else "Unknown Answer"

        
        # Tokenize the target answer as labels
        labels_enc = self.tokenizer(
            target_answer,
            max_length=self.config["max_target_length"],
            truncation=True,
            return_tensors="pt"
        )
        labels = labels_enc.input_ids[0]  # [answer_len]
        
        return {
            "inputs_embeds": combined_inputs_embeds,
            "attention_mask": combined_attention_mask,
            "labels": labels,
            "question": question,
            "all_answers": answers
        }



######################
# Collate Function
######################
def collate_fn(batch):
    # Find max seq lengths for inputs and labels
    max_seq_len = max(b["inputs_embeds"].size(0) for b in batch)
    max_label_len = max(b["labels"].size(0) for b in batch)

    inputs_embeds = torch.stack([
        F.pad(b["inputs_embeds"], (0, 0, 0, max_seq_len - b["inputs_embeds"].size(0)), value=0.0)
        for b in batch
    ])
    attention_mask = torch.stack([
        F.pad(b["attention_mask"], (0, max_seq_len - b["attention_mask"].size(0)), value=0)
        for b in batch
    ])
    labels = torch.stack([
        F.pad(b["labels"], (0, max_label_len - b["labels"].size(0)), value=-100)
        for b in batch
    ])

    questions = [b["question"] for b in batch]
    all_answers = [b["all_answers"] for b in batch]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
        "questions": questions,
        "all_answers": all_answers
    }


######################
# Model Wrapper
######################
class VT5Model(torch.nn.Module):
    def __init__(self, vt5_model_path):
        super(VT5Model, self).__init__()
        self.vt5_decoder = T5ForConditionalGeneration.from_pretrained(vt5_model_path)
    
    def parallelize(self):
        self.vt5_decoder = nn.DataParallel(self.vt5_decoder)

    def forward(self, inputs_embeds, attention_mask, labels=None, return_pred_answer=False, tokenizer=None):
        # Directly pass inputs_embeds to the model as encoder inputs.
        # labels are used to compute loss.
        outputs = self.vt5_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        if return_pred_answer and tokenizer is not None:
            pred_ids = self.vt5_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            pred_answers = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return outputs, pred_answers

        return outputs


######################
# Training & Evaluation
######################
def train_epoch(data_loader, model, optimizer, scheduler, device, logger):
    model.train()
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

        outputs = model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_train_metrics(loss=loss.item(), lr=current_lr, step=batch_idx)


def evaluate(data_loader, model, tokenizer, evaluator, device):
    model.eval()
    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", unit="batch")):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

            # Generate predictions
            pred_ids = self.vt5_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            pred_answers = tokenizer.batch_decode(pred_ids['sequences'], skip_special_tokens=True)
            pred_answers_conf = model_utils.get_generative_confidence(pred_ids)  # Optional confidence extraction

            refs = batch["all_answers"]

            all_preds.extend(preds)
            all_refs.extend(refs)

    metrics = evaluator.get_metrics(all_refs, all_preds)
    mean_accuracy = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
    mean_anls = np.mean(metrics['anls']) if metrics['anls'] else 0
    return mean_accuracy, mean_anls


######################
# Main
######################
def main():
    # Load configuration
    config = load_config("config/models/vt5.yml")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # Initialize model and tokenizer
    model_path = config["model_weights"]
    model = VT5Model(model_path).to(device)

    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and config.get("data_parallel", False):
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training.")
        model.parallelize()

    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # Paths
    train_embeddings_dir = config["dataset_paths"]["train_embeddings_dir"]
    val_embeddings_dir = config["dataset_paths"]["val_embeddings_dir"]
    train_qas_path = config["dataset_paths"]["train_qas_path"]
    val_qas_path = config["dataset_paths"]["val_qas_path"]

    # Datasets and Dataloaders
    train_dataset = PrecomputedDocVQADataset(
        embeddings_dir=train_embeddings_dir,
        qas_json_path=train_qas_path,
        tokenizer=tokenizer,
        config=config,
        model=model
    )
    val_dataset = PrecomputedDocVQADataset(
        embeddings_dir=val_embeddings_dir,
        qas_json_path=val_qas_path,
        tokenizer=tokenizer,
        config=config,
        model=model
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    # Logger and Evaluator
    logger = Logger(config)
    evaluator = Evaluator()

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=float(config["training_parameters"]["lr"]))
    total_steps = config["training_parameters"]["train_epochs"] * len(train_loader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=config["training_parameters"]["warmup_iterations"],
        num_training_steps=total_steps
    )

    best_accuracy = 0.0
    for epoch in range(config["training_parameters"]["train_epochs"]):
        print(f"Starting epoch {epoch}...")
        train_epoch(train_loader, model, optimizer, scheduler, device, logger)
        accuracy, anls = evaluate(val_loader, model, tokenizer, evaluator, device)

        print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}, ANLS: {anls:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.vt5_decoder.module.save_pretrained(config["save_dir"])
            tokenizer.save_pretrained(config["save_dir"])

        # Clear GPU memory after epoch
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    main()
