#!/usr/bin/env python3
import os
import sys
import json
import random
import gc
import torch
# Add the parent directory of the current script to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_scheduler
)

# Local imports
from logger import Logger
from metrics import Evaluator
import layoutlmft.models.vqa._model_utils as model_utils  # for generative confidence if needed
from layoutlmft.models.vqa._modules import CustomT5Config, SpatialEmbeddings, VisualEmbeddings
from layoutlmft.models.vqa.vt5 import ProxyVT5  # Importing ProxyVT5 from vt5.py
####################
#problems: added the decoded id
#############
###############################################################################
# 1) Utility: Load YAML Config
###############################################################################
def load_config(config_file):
    import yaml
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


###############################################################################
# 2) Hybrid Model: Freeze VT5 encoder, swap fresh T5-base decoder
###############################################################################
class HybridVT5GraphDoc:
    """
    Combines ProxyVT5 with additional document embeddings.
    Steps:
    1. Use ProxyVT5 to prepare input embeddings (question + OCR + image).
    2. Pass input embeddings through the frozen encoder to get question_enc.
    3. Concatenate precomputed doc_embeds.
    4. Pass the combined embeddings to the fresh T5-base decoder.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

        # 1) Initialize ProxyVT5
        self.vt5 = ProxyVT5(config)

        # 2) Freeze the VT5 encoder
        # for param in self.vt5.model.encoder.parameters():
        for param in self.vt5.model.parameters():
            param.requires_grad = False

        # 3) Load a fresh T5-base model and extract its decoder
        print("[HybridVT5GraphDoc] Loading fresh T5-base decoder.")
        base_config = T5Config.from_pretrained("t5-base")
        base_full = T5ForConditionalGeneration.from_pretrained("t5-base", config=base_config)
        fresh_decoder = base_full.decoder

        # 4) Replace the original decoder with the fresh T5-base decoder
        print("[HybridVT5GraphDoc] Replacing original decoder with fresh T5-base decoder.")
        self.vt5.model.decoder = fresh_decoder

        # 5) Keep the same tokenizer from ProxyVT5
        self.tokenizer = self.vt5.tokenizer

        # 6) Move the model to the specified device
        self.vt5.model.to(self.device)
        if torch.cuda.device_count() > 1 and config.get("data_parallel", False):
            self.vt5.model = torch.nn.DataParallel(self.vt5.model)

    def forward(self, batch, return_pred_answer=False, return_confidence=False):
        """
        Forward pass through the hybrid model.
        """
        # A) Prepare input embeddings using ProxyVT5
        input_embeds, attention_mask, labels = self.vt5.prepare_inputs_for_vqa(
            batch["questions"],       # List of strings
            batch["words"],           # List of lists of strings
            batch["boxes"],           # List of lists of bounding boxes [x1, y1, x2, y2]
            batch["images"],          # List of image tensors or None
            batch["all_answers"]      # List of lists of strings
        )

        # B) Pass input embeddings through the frozen encoder
        encoder_outputs = self.vt5.model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        question_enc = encoder_outputs.last_hidden_state  # [B, seq_len, d_model]

        print(f"Question Embeddings VT5 Shape (should be both Q AND Context): {question_enc.shape}") 

        # C) Concatenate precomputed doc_embeds
        doc_embeds_list = batch["doc_embeds"]  # List of tensors [seq_len, d_model]
        B = len(doc_embeds_list)
        d_model = question_enc.size(-1)
        max_doc_len = max(e.size(0) for e in doc_embeds_list)
        padded_doc_embeds = torch.zeros(B, max_doc_len, d_model, device=self.device)
        padded_doc_mask = torch.zeros(B, max_doc_len, dtype=torch.long, device=self.device)
        # Fill padded tensors with actual data
        for i in range(B):
            seq_len = doc_embeds_list[i].size(0)
            padded_doc_embeds[i, :seq_len, :] = doc_embeds_list[i].to(self.device)
            padded_doc_mask[i, :seq_len] = batch["doc_masks"][i].to(self.device)

        for i in range(B):
            emb = doc_embeds_list[i].to(self.device)
            padded_doc_embeds[i, :emb.size(0), :] = emb
        
        print(f"Document Embeddings Padded Shape: {padded_doc_embeds.shape}") 

        final_enc = torch.cat([question_enc, padded_doc_embeds], dim=1)  # [B, seq_len + doc_seq_len, d_model]

        # Concatenate attention masks
        final_attention_mask = torch.cat([attention_mask, padded_doc_mask], dim=1) 
         # Print the shape of the concatenated embeddings
        print(f"Final Concatenated Encodings Shape: {final_enc.shape}")  # Example: [B, seq_len + doc_seq_len, d_model]

        # D) Pass combined embeddings to the decoder
        outputs = self.vt5.model(
            inputs_embeds=final_enc,
            labels=labels,
            return_dict=True
        )

        # E) Generate predictions if required
        batch_size = final_enc.size(0)  # Determine batch size from `final_enc`
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.tokenizer.pad_token_id

        # Debugging prints for `generate` inputs
        print(f"[Debug] Before generate call:")
        print(f"  final_enc.shape: {final_enc.shape}")
        print("padded doc shape: ", {padded_doc_embeds.shape})
        print(f"  final_enc.device: {final_enc.device}")
        print(f"  final_enc.max(): {final_enc.max()}, final_enc.min(): {final_enc.min()}")
        print(f"  decoder_input_ids.shape: {decoder_input_ids.shape}")
        print(f"  decoder_input_ids.device: {decoder_input_ids.device}")
        print(f"  decoder_input_ids.max(): {decoder_input_ids.max()}, decoder_input_ids.min(): {decoder_input_ids.min()}")

        if attention_mask is not None:
            print(f"  attention_mask.shape: {attention_mask.shape}")
            print(f"  attention_mask.device: {attention_mask.device}")
            print(f"  attention_mask.max(): {attention_mask.max()}, attention_mask.min(): {attention_mask.min()}")
        else:
            print(f"  attention_mask: None")
        print(f"padded_doc_mask: {padded_doc_mask}")
        print(f"final_attention_mask: {final_attention_mask}")
        print(f"final_enc.shape: {final_enc.shape}")

        pred_answers = None
        pred_conf = None
        if return_pred_answer:
            with torch.no_grad():
                gen_out = self.vt5.model.generate(
                    inputs_embeds=final_enc,
                    attention_mask=final_attention_mask,
                    output_scores=return_confidence,
                    return_dict_in_generate=return_confidence,
                    decoder_input_ids=decoder_input_ids,
                )
                if return_confidence:
                    seqs = gen_out["sequences"]
                else:
                    seqs = gen_out

                pred_answers = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

                if return_confidence:
                    # Implement confidence extraction if needed
                    pred_conf = model_utils.get_generative_confidence(gen_out)
        
        return outputs, pred_answers, pred_conf


###############################################################################
# 3) Dataset to load OCR tokens/boxes + doc embeddings + images
###############################################################################
class DocVQAQuestionDrivenDataset(Dataset):
    """
    1) Q&A from spdocvqa_qas/<split>_v1.0_withQT.json
    2) OCR from spdocvqa_ocr/<image_stem>.json
    3) .pt doc embeddings from spdocvqa_embeddings/<image_stem>.pt
    4) images from spdocvqa_images/<image_stem>.png
    """

    def __init__(self, config, split):
        super().__init__()
        self.config = config
        self.split = split

        # Set paths based on split
        if split == "train":
            qa_json_path = config["dataset_paths"]["train_qas_path"]
            self.embeddings_dir = config["dataset_paths"]["train_embeddings_dir"]
        else:
            qa_json_path = config["dataset_paths"]["val_qas_path"]
            self.embeddings_dir = config["dataset_paths"]["val_embeddings_dir"]

        self.ocr_dir = config["dataset_paths"].get("ocr_dir", None)
        self.images_dir = config["dataset_paths"].get("images_dir", None)

        # Load Q&A JSON
        with open(qa_json_path, "r") as f:
            data = json.load(f)
        self.samples_raw = data["data"]

        # Filter samples that have corresponding .pt embeddings
        self.valid_samples = []
        none_answer_count = 0  # Counter for items with "none" as the answer

        for item in self.samples_raw:
            image_stem = Path(item["image"]).stem
            pt_file = Path(self.embeddings_dir) / f"{image_stem}.pt"
            if pt_file.exists():
                answers = item.get("answers", ["none"])
                if answers == ["none"]:
                    none_answer_count += 1  # Increment the counter for "none" answers
                
                self.valid_samples.append({
                    "image_stem": image_stem,
                    "question": item["question"],
                    "answers": answers,
                })

        # Print the count of "none" answers
        print(f"Number of items with 'none' as the answer: {none_answer_count}")

        print(f"[DocVQAQuestionDrivenDataset] Found {len(self.valid_samples)} valid items for split='{split}'")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        data_item = self.valid_samples[idx]
        image_stem = data_item["image_stem"]
        question_text = data_item["question"]
        answers_list = data_item["answers"]

        # Load .pt doc embeddings
        pt_path = Path(self.embeddings_dir) / f"{image_stem}.pt"
        doc_data = torch.load(pt_path)
        doc_embeds = doc_data["last_hidden_state"].squeeze(0)  # [seq_len, d_model]
        doc_mask = doc_data["attention_mask"].squeeze(0)      # [seq_len]

        # Load OCR data
        words, boxes = self.load_ocr_data(image_stem)

        # Convert Microsoft OCR bbox format [x1, y1, x2, y2, x3, y3, x4, y4] to [x1, y1, x3, y3]
        converted_boxes = []
        unexpected_format_count = 0  # Counter for unexpected formats
        for bbox in boxes:
            if len(bbox) == 8:
                x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                converted_box = [x1, y1, x3, y3]
            elif len(bbox) == 4:
                # If bbox is already in [x1, y1, x2, y2] format
                converted_box = bbox
            else:
                # Handle unexpected bbox formats
                converted_box = [0, 0, 0, 0]
                unexpected_format_count += 1  
            converted_boxes.append(converted_box)
        # Debugging: Print some sample converted boxes
        if idx < 5:  # Limit to first 5 samples for brevity
            print(f"Sample {idx}: Original bbox length={len(boxes[idx])}, Converted bbox={converted_boxes[idx]}")
        
        print(f"Number of unexpected bounding box formats: {unexpected_format_count}")

        # Load Image
        image_tensor = None
        if self.images_dir:
            img_path = Path(self.images_dir) / f"{image_stem}.png"
            if img_path.exists():
                pil_img = Image.open(img_path).convert("RGB")
                image_tensor = pil_img  # Pass raw PIL image (no resizing)
            else:
                print(f"[Warning] Image not found: {img_path}")
                # Optionally handle missing images

        # Choose random target answer
        target_answer = random.choice(answers_list) if answers_list else "none"

        return {
            "questions": [question_text],          # List of one question
            "all_answers": [answers_list],        # List of answers lists
            "target_answers": [target_answer],    # List of one target answer
            "doc_embeds": [doc_embeds],           # List of one doc embedding tensor
            "doc_masks": [doc_mask],              # List of one doc mask tensor
            "words": [words],                      # List of lists of strings
            "boxes": [converted_boxes],            # List of lists of converted bounding boxes [x1, y1, x2, y2]
            "images": [image_tensor]               # List of one image tensor or None
        }

    def load_ocr_data(self, stem):
        """
        Reads spdocvqa_ocr/<stem>.json and extracts words and bounding boxes.
        Returns (list_of_words, list_of_bboxes)
        """
        if not self.ocr_dir:
            return [], []

        ocr_path = Path(self.ocr_dir) / f"{stem}.json"
        if not ocr_path.exists():
            print(f"[Warning] OCR file not found: {ocr_path}")
            return [], []

        with open(ocr_path, "r") as f:
            ocr_data = json.load(f)

        recognition_results = ocr_data.get("recognitionResults", [])
        all_words = []
        all_boxes = []
        for page_info in recognition_results:
            lines = page_info.get("lines", [])
            for line in lines:
                words = line.get("words", [])
                for word in words:
                    all_words.append(word.get("text", ""))
                    all_boxes.append(word.get("boundingBox", [0, 0, 0, 0, 0, 0, 0, 0]))
        return all_words, all_boxes


###############################################################################
# 4) Collate Function (Defined Outside the Dataset Class)
###############################################################################
def docvqa_collate_fn(batch):
    """
    Collate function to combine batch items into tensors.
    """
    questions = [b["questions"][0] for b in batch]
    all_answers = [b["all_answers"][0] for b in batch]
    target_answers = [b["target_answers"][0] for b in batch]
    doc_embeds = [b["doc_embeds"][0] for b in batch]
    doc_masks = [b["doc_masks"][0] for b in batch]
    words = [b["words"][0] for b in batch]
    boxes = [b["boxes"][0] for b in batch]
    images = [b["images"][0] for b in batch]

    # After processing boxes
    print("Batch bbox shapes and sample values:")
    for i, box in enumerate(boxes):
        print(f"  Sample {i}: Number of boxes={len(box)}, First bbox={box[0] if len(box) > 0 else 'N/A'}")
    


    # Handle images: convert PIL Images to tensors if they are not None
    processed_images = []
    for img in images:
        if img is not None:
            preprocess = transforms.ToTensor()
            img_tensor = preprocess(img)
            processed_images.append(img_tensor)
        else:
            # Create a dummy tensor if image is None
            processed_images.append(torch.zeros(3, 224, 224))  # Example size

    return {
        "questions": questions,            # List of strings
        "all_answers": all_answers,        # List of lists of strings
        "target_answers": target_answers,  # List of strings
        "doc_embeds": doc_embeds,          # List of tensors [seq_len, d_model]
        "doc_masks": doc_masks,            # List of tensors [seq_len]
        "words": words,                    # List of lists of strings
        "boxes": boxes,                    # List of lists of bounding boxes [x1, y1, x2, y2]
        "images": processed_images          # List of tensors [3, H, W]
    }


###############################################################################
# 5) Training / Evaluation
###############################################################################
def train_epoch(data_loader, model_wrapper, optimizer, scheduler, logger=None, evaluator=None, device=None):
    model_wrapper.vt5.model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Training", unit="batch")):
        # Debugging input batch
        print("Debugging batch before forward pass:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, device={value.device}, max={value.max()}, min={value.min()}")

        outputs, pred_answers, _ = model_wrapper.forward(batch, return_pred_answer=True)

        # Debugging outputs
        print("Debugging outputs after forward pass:")
        if outputs.loss is not None:
            print(f"  Loss value: {outputs.loss.item()}")
        if outputs.logits is not None:
            print(f"  Logits shape: {outputs.logits.shape}")

        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Evaluate predictions for immediate feedback
        gt_answers = batch["all_answers"]
        metric = evaluator.get_metrics(gt_answers, pred_answers)
        batch_acc = float(np.mean(metric["accuracy"])) if metric["accuracy"] else 0.0
        batch_anls = float(np.mean(metric["anls"])) if metric["anls"] else 0.0

        if logger is not None:
            current_step = logger.current_epoch * len(data_loader) + step
            logger.log_train_metrics(loss.item(), optimizer.param_groups[0]["lr"], current_step)
            logger.writer.add_scalar("Train/Batch Accuracy", batch_acc, current_step)
            logger.writer.add_scalar("Train/Batch ANLS", batch_anls, current_step)

        if step % 20 == 0:
            print(f" Step={step}, Loss={loss.item():.4f}, Acc={batch_acc:.4f}, ANLS={batch_anls:.4f}")

    avg_loss = total_loss / len(data_loader)
    if logger is not None:
        logger.writer.add_scalar("Train/Epoch Loss", avg_loss, logger.current_epoch)
    return avg_loss


def evaluate(data_loader, model_wrapper, logger=None, evaluator=None, device=None, return_confidence=False):
    model_wrapper.vt5.model.eval()
    all_preds = []
    all_refs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            outputs, preds, conf = model_wrapper.forward(batch, return_pred_answer=True, return_confidence=return_confidence)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item()

            all_preds.extend(preds)
            all_refs.extend(batch["all_answers"])

    metric = evaluator.get_metrics(all_refs, all_preds)
    mean_acc = float(np.mean(metric["accuracy"])) if metric["accuracy"] else 0.0
    mean_anls = float(np.mean(metric["anls"])) if metric["anls"] else 0.0
    avg_loss = total_loss / len(data_loader)

    print(f"[Val] Loss={avg_loss:.4f}, Acc={mean_acc:.4f}, ANLS={mean_anls:.4f}")
    if logger is not None:
        logger.writer.add_scalar("Validation/Loss", avg_loss, logger.current_epoch)
        logger.writer.add_scalar("Validation/Accuracy", mean_acc, logger.current_epoch)
        logger.writer.add_scalar("Validation/ANLS", mean_anls, logger.current_epoch)

    return mean_acc, mean_anls


###############################################################################
# 6) Main
###############################################################################
def main():
    # A) Load config
    config_file = "config/models/vt5.yml"
    config = load_config(config_file)

    # B) Initialize logger and evaluator
    logger = Logger(config)
    evaluator = Evaluator()

    # C) Build datasets and dataloaders
    train_dataset = DocVQAQuestionDrivenDataset(config, split="train")
    val_dataset = DocVQAQuestionDrivenDataset(config, split="val")

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

    # D) Initialize model
    hybrid_model = HybridVT5GraphDoc(config)

    # E) Set up optimizer and scheduler
    optimizer = AdamW(hybrid_model.vt5.model.parameters(), lr=float(config["training_parameters"]["lr"]))
    total_steps = config["training_parameters"]["train_epochs"] * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config["training_parameters"]["warmup_iterations"],
        num_training_steps=total_steps
    )

    # F) Training loop
    best_acc = 0.0
    for epoch in range(config["training_parameters"]["train_epochs"]):
        print(f"\n=== Starting epoch {epoch+1}/{config['training_parameters']['train_epochs']} ===")
        logger.current_epoch = epoch

        # Train
        train_loss = train_epoch(train_loader, hybrid_model, optimizer, scheduler, logger, evaluator)
        print(f" [Epoch {epoch+1}] Training Loss: {train_loss:.4f}")

        # Validate
        acc, anls = evaluate(val_loader, hybrid_model, logger, evaluator)
        print(f" [Epoch {epoch+1}] Validation Acc: {acc:.4f}, ANLS: {anls:.4f}")

        # Save the best model
        if acc > best_acc:
            best_acc = acc
            print(f"[Info] New best model at epoch {epoch+1} with Acc={acc:.4f}. Saving model...")
            save_dir = Path(config["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(hybrid_model.vt5.model, "module"):
                # If using DataParallel
                hybrid_model.vt5.model.module.save_pretrained(save_dir)
            else:
                hybrid_model.vt5.model.save_pretrained(save_dir)
            hybrid_model.tokenizer.save_pretrained(save_dir)
            print(f"[Info] Model saved to {save_dir}")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[Info] Training complete. Best Validation Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
