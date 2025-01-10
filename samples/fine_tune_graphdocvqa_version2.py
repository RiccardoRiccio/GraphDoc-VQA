import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import json
import random
import gc
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_scheduler
)
from torch.optim import AdamW

# Local imports (assuming logger.py, metrics.py, model_utils.py exist in your PYTHONPATH)
from logger import Logger
from metrics import Evaluator
import layoutlmft.models.vqa._model_utils as model_utils


######################################################################
# 1) Utility: Load YAML Config
######################################################################
def load_config(config_file):
    """Utility to load a YAML config file."""
    import yaml
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

######################################################################
# 2) Dataset
######################################################################
class DocVQAQuestionDrivenDataset(Dataset):
    """
    Similar to 'SPDocVQA' or your original 'DocVQAQuestionDrivenDataset'.
    We have:
      - "data": [ { "questionId":..., "question":..., "answers": [...], "image": ... }, ... ]
    For each item:
      - We load the .pt doc embedding that matches the image stem.
      - We pick 1 random answer (for training).
      - We keep all answers for evaluation.
    """
    def __init__(self, embeddings_dir, qa_json_path, tokenizer, config):
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.tokenizer = tokenizer
        self.config = config
        self.max_source_length = config.get("max_source_length", 512)

        # Load the JSON that has {"data": [...]}
        with open(qa_json_path, "r") as f:
            full_data = json.load(f)

        self.samples_raw = full_data["data"]  # list of dicts
        self.valid_samples = []

        # Build the valid samples that actually have .pt embeddings
        for item in self.samples_raw:
            image_path = item["image"]   # e.g., "documents/xnbl0037_1.png"
            image_stem = Path(image_path).stem  # "xnbl0037_1"

            pt_file = Path(self.embeddings_dir) / f"{image_stem}.pt"
            if not pt_file.exists():
                # print(f"[Warning] No precomputed embeddings for {pt_file}, skipping.")
                continue

            self.valid_samples.append({
                "question": item["question"],
                "answers": item.get("answers", ["none"]),
                "pt_file": str(pt_file)
            })

        print(f"[DocVQAQuestionDrivenDataset] Found {len(self.valid_samples)} valid items.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        # Read the sample
        data_item = self.valid_samples[idx]
        question_text = data_item["question"]
        answers_list  = data_item["answers"]
        pt_file       = data_item["pt_file"]

        # Load doc embeddings
        doc_data  = torch.load(pt_file)  # e.g. { "last_hidden_state":..., "attention_mask":... }
        doc_embeds = doc_data["last_hidden_state"].squeeze(0)   # shape: [doc_seq_len, hidden_dim]
        doc_mask   = doc_data["attention_mask"].squeeze(0)      # shape: [doc_seq_len]

        # Pick one random answer for training
        if answers_list:
            target_answer = random.choice(answers_list)
        else:
            target_answer = "none"

        # Return everything. We'll do question tokenization + doc embedding 
        # concatenation in the model or collate function if we want. 
        return {
            "question_text": question_text,
            "answers": answers_list,
            "target_answer": target_answer,
            "doc_embeds": doc_embeds,
            "doc_mask": doc_mask
        }


######################################################################
# 3) Collate Function
######################################################################
def docvqa_collate_fn(batch):
    """
    We convert a list of items -> batch dict. 
    We do minimal processing: just group them. 
    The model or training loop can do further padding or tokenization as needed.
    """
    questions      = [b["question_text"] for b in batch]
    all_answers    = [b["answers"]       for b in batch]
    target_answers = [b["target_answer"] for b in batch]
    doc_embeds     = [b["doc_embeds"]    for b in batch]
    doc_masks      = [b["doc_mask"]      for b in batch]

    return {
        "questions": questions,
        "all_answers": all_answers,
        "target_answers": target_answers,
        "doc_embeds": doc_embeds,
        "doc_masks": doc_masks
    }

######################################################################
# 4) Model: "ProxyVT5Precomputed"
######################################################################
class ProxyVT5Precomputed:
    """
    Adapts the approach of 'ProxyVT5', but 
    1) Skips bounding-box + visual embeddings (since doc is precomputed)
    2) Only encodes question
    3) Concat question + doc embeddings
    4) T5 decodes to predict an answer
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(config["model_weights"])
        self.model = T5ForConditionalGeneration.from_pretrained(config["model_weights"])

        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        self.batch_size = config["batch_size"]
        self.max_source_length = config.get("max_source_length", 512)

        # If data_parallel is true and multiple GPUs available
        if torch.cuda.device_count() > 1 and config.get("data_parallel", False):
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

    def prepare_inputs_for_vqa(self, questions, doc_embeds_list, doc_masks_list, target_answers=None):
        """
        1) Tokenize question 
        2) Convert question to T5 embeddings
        3) Pad doc_embeds
        4) Concat question + doc
        5) If training, tokenize answers -> labels 
        6) Return (input_embeds, attention_mask, labels)
        """
        B = len(questions)

        # (1) Tokenize question
        tokens = self.tokenizer(
            ["question: " + q for q in questions],
            max_length=self.max_source_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        )
        q_input_ids = tokens.input_ids.to(self.device)        # [B, Q_len]
        q_mask      = tokens.attention_mask.to(self.device)   # [B, Q_len]

        # (2) Convert question input_ids -> T5 embeddings
        #     The T5 model has a shared embedding layer: self.model.shared
        q_embeds = self.model.module.shared(q_input_ids) # [B, Q_len, d_model]

        # (3) Pad doc_embeds
        d_model = q_embeds.size(-1)
        max_doc_len = max(e.size(0) for e in doc_embeds_list)  # doc_embeds_list[i] -> [doc_seq_len, d_model]
        padded_doc_embeds = torch.zeros(B, max_doc_len, d_model, device=self.device)
        padded_doc_mask   = torch.zeros(B, max_doc_len, dtype=torch.long, device=self.device)

        for i in range(B):
            seq_len = doc_embeds_list[i].size(0)
            padded_doc_embeds[i, :seq_len, :] = doc_embeds_list[i].to(self.device)
            padded_doc_mask[i, :seq_len]      = doc_masks_list[i].to(self.device)

        # (4) Concat question + doc
        input_embeds   = torch.cat([q_embeds, padded_doc_embeds], dim=1)  # [B, Q_len + doc_len, d_model]
        attention_mask = torch.cat([q_mask,   padded_doc_mask], dim=1)    # [B, Q_len + doc_len]

        # (5) Build labels from target_answers if we have them
        labels = None
        if target_answers is not None:
            labels_enc = self.tokenizer(
                target_answers,
                max_length=self.max_source_length,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            labels = labels_enc.input_ids  # [B, L]
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels = labels.to(self.device)

        return input_embeds, attention_mask, labels

    def forward(self, batch, return_pred_answer=False, return_confidence=True):
        """
        1) Prepare inputs
        2) Forward pass
        3) Optionally generate predictions (with or without confidence)
        """
        questions      = batch["questions"]
        doc_embeds_list= batch["doc_embeds"]
        doc_masks_list = batch["doc_masks"]
        target_answers = batch["target_answers"]

        # Prepare
        input_embeds, attention_mask, labels = self.prepare_inputs_for_vqa(
            questions, doc_embeds_list, doc_masks_list, target_answers
        )

        # Forward pass
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Optionally generate

        # print("start id token", self.model.module.config.decoder_start_token_id)
        pred_answers = None
        pred_answers_conf = None


        batch_size = input_embeds.size(0) 
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.tokenizer.pad_token_id

        if return_pred_answer:
            with torch.no_grad():
                gen_out = self.model.module.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_length=32,
                    num_beams=4,
                    output_scores=return_confidence,
                    return_dict_in_generate=return_confidence,
                    decoder_input_ids=decoder_input_ids,
                )


                pred_answers = self.tokenizer.batch_decode(gen_out["sequences"], skip_special_tokens=True)

                if return_confidence:
                    # If you have a function like model_utils.get_generative_confidence(...)
                    pred_answers_conf = model_utils.get_generative_confidence(gen_out)

        return outputs, pred_answers, pred_answers_conf


######################################################################
# 5) Training
######################################################################
# def train_epoch(data_loader, model_wrapper, optimizer, scheduler, logger=None, evaluator=None, device=None):
#     """
#     Single training epoch. 
#     """
#     model_wrapper.model.train()
#     total_loss = 0

#     # We'll track iteration steps for logging
#     for step, batch in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
#         # If you want to ensure we push batch to device, do so. 
#         # But in this case, we do it in the model's forward pass.

#         outputs, _, _ = model_wrapper.forward(batch, return_pred_answer=False)
#         loss = outputs.loss
#         # print(f"Loss: {loss}, Shape: {loss.shape}")  
#         # loss.backward()
#         # Debugging
#         # print(f"Loss: {loss}, Shape: {loss.shape}")

#         # Reduce loss to scalar if needed
#         if loss.dim() > 0:
#             loss = loss.mean()
#             # print("mean loss", loss)  # Average across GPUs (works for batch size 1)

#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()

#         total_loss += loss.item()

#         # Logging
#         if logger is not None:
#             current_step = logger.current_epoch * len(data_loader) + step
#             logger.log_train_metrics(loss.item(), optimizer.param_groups[0]["lr"], current_step)


#         if step % 20 == 0:
#             print(f" Step={step}, Loss={loss.item():.4f}")

#     avg_loss = total_loss / len(data_loader)
#     # print(f"[Train] Average loss: {avg_loss:.4f}")
#     if logger is not None:
#         logger.writer.add_scalar("Train/Epoch Loss", avg_loss, logger.current_epoch)
def train_epoch(data_loader, model_wrapper, optimizer, scheduler, logger=None, evaluator=None, device=None):
    model_wrapper.model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        outputs, pred_answers, _ = model_wrapper.forward(batch, return_pred_answer=True)
        loss = outputs.loss

        # Evaluate batch predictions against ground truth
        gt_answers = batch['all_answers']
        metric = evaluator.get_metrics(gt_answers, pred_answers)
        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])

        # print(f"Step={step}, Prediction: {pred_answers}, Ground Truth: {gt_answers}, Batch ANLS: {batch_anls:.4f}")


        if loss.dim() > 0:
            loss = loss.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Log batch-level metrics (Fixed)
        if logger is not None:
            current_step = logger.current_epoch * len(data_loader) + step
            logger.log_train_metrics(loss.item(), optimizer.param_groups[0]["lr"], current_step)
            logger.writer.add_scalar("Train/Batch Accuracy", batch_acc, current_step)
            logger.writer.add_scalar("Train/Batch ANLS", batch_anls, current_step)

 # Fixed line

        if step % 20 == 0:
            print(f" Step={step}, Loss={loss.item():.4f}, Accuracy={batch_acc:.4f}, ANLS={batch_anls:.4f}")

    avg_loss = total_loss / len(data_loader)
    if logger is not None:
        logger.writer.add_scalar("Train/Epoch Loss", avg_loss, logger.current_epoch)



def evaluate(data_loader, model_wrapper, logger=None, evaluator=None, device=None, return_confidence=False):
    """
    Evaluate on val data. We'll compute metrics using evaluator if present.
    """
    model_wrapper.model.eval()

    all_preds = []
    all_refs  = []
    all_confs = []

    with torch.no_grad():
        for batch in data_loader:
            _, preds, confs = model_wrapper.forward(batch, return_pred_answer=True, return_confidence=return_confidence)
            all_preds.extend(preds)
            all_refs.extend(batch["all_answers"])
            if confs is not None:
                all_confs.extend(confs)

    # Evaluate
    if evaluator is not None:
        metric = evaluator.get_metrics(all_refs, all_preds)
        mean_acc  = float(np.mean(metric["accuracy"])) if len(metric["accuracy"])>0 else 0
        mean_anls = float(np.mean(metric["anls"])) if len(metric["anls"])>0 else 0
    else:
        # fallback: simple exact match
        correct = 0
        total   = len(all_preds)
        for pred, refs in zip(all_preds, all_refs):
            if pred in refs:
                correct += 1
        mean_acc  = correct / total if total>0 else 0
        mean_anls = 0

    print(f"[Validation] Accuracy={mean_acc:.4f}, ANLS={mean_anls:.4f}")

    # Confidence
    if return_confidence and len(all_confs) == len(all_preds):
        mean_conf = sum(all_confs) / len(all_confs)
        print(f"[Validation] Average confidence={mean_conf:.4f}")

    # Logging
    if logger is not None:
        logger.writer.add_scalar("Validation/Accuracy", mean_acc, logger.current_epoch)
        logger.writer.add_scalar("Validation/ANLS", mean_anls, logger.current_epoch)

    return mean_acc, mean_anls


######################################################################
# 6) Main
######################################################################
def main():
    # 1) Load config
    config = load_config("config/models/vt5.yml")  # or wherever your config file is
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 2) Prepare saving dir
    os.makedirs(config["save_dir"], exist_ok=True)

    # 3) Create logger & evaluator
    logger = Logger(config)
    evaluator = Evaluator()

    # 4) Build Model
    vt5_model = ProxyVT5Precomputed(config)

    # 5) Dataset + Dataloaders
    train_dataset = DocVQAQuestionDrivenDataset(
        embeddings_dir=config["dataset_paths"]["train_embeddings_dir"],
        qa_json_path=config["dataset_paths"]["train_qas_path"],
        tokenizer=vt5_model.tokenizer,
        config=config
    )
    val_dataset = DocVQAQuestionDrivenDataset(
        embeddings_dir=config["dataset_paths"]["val_embeddings_dir"],
        qa_json_path=config["dataset_paths"]["val_qas_path"],
        tokenizer=vt5_model.tokenizer,
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=docvqa_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=docvqa_collate_fn
    )

    # 6) Optimizer & Scheduler
    optimizer = AdamW(vt5_model.model.parameters(), lr=float(config["training_parameters"]["lr"]))
    total_steps = config["training_parameters"]["train_epochs"] * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config["training_parameters"]["warmup_iterations"],
        num_training_steps=total_steps
    )

    # 7) Train/Eval Loop
    best_accuracy = 0.0
    for epoch in range(config["training_parameters"]["train_epochs"]):
        print(f"\n=== Starting epoch {epoch} ===")
        logger.current_epoch = epoch

        # Train
        train_epoch(
            data_loader=train_loader,
            model_wrapper=vt5_model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            evaluator=evaluator,
            device=device
        )

        # Validate
        accuracy, anls = evaluate(
            data_loader=val_loader,
            model_wrapper=vt5_model,
            logger=logger,
            evaluator=evaluator,
            device=device,
            return_confidence=True  # if you want confidence
        )

        # Save if best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"[Info] Found new best model at epoch {epoch}, accuracy={accuracy:.4f}. Saving...")
            vt5_model.model.module.save_pretrained(config["save_dir"])
            vt5_model.tokenizer.save_pretrained(config["save_dir"])

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
