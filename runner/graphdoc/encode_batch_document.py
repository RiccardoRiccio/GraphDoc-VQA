#########################
# SAME CODE AS encode_document.py BUT HERE WE CAN SEND BATCHES OF IMAGE-QUESTIONS PAIR
########################


import cv2
import json
import torch
import numpy as np
import sys
sys.path.append('./')

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from transformers import AutoModel, AutoTokenizer

# Function to read OCR data
def read_ocr(json_path):
    """Read OCR data and extract line-level information."""
    ocr_data = json.load(open(json_path, 'r'))  # Load JSON
    polys = []  # Store bounding boxes for each line
    contents = []  # Store line texts

    # Navigate the JSON hierarchy and extract line-level data
    for page in ocr_data['recognitionResults']:
        for line in page['lines']:
            contents.append(line['text'])  # Extract line text
            # Convert the bounding box of the line to the required format
            bbox = [[line['boundingBox'][i], line['boundingBox'][i + 1]] for i in range(0, len(line['boundingBox']), 2)]
            polys.append(bbox)  # Append bounding box for the line

    return polys, contents

# Convert polys to bounding boxes
def polys2bboxes(polys):
    bboxes = []
    for poly in polys:
        poly = np.array(poly).reshape(-1)
        x1 = poly[0::2].min()
        y1 = poly[1::2].min()
        x2 = poly[0::2].max()
        y2 = poly[1::2].max()
        bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes).astype('int64')
    return bboxes

# Mean pooling for embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Extract sentence embeddings
def extract_sentence_embeddings(contents, tokenizer, sentence_bert):
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return sentence_embeddings

# Process question embedding
def process_question(question, tokenizer, sentence_bert):
    encoded_input = tokenizer([question], padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    question_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return question_embedding[0]

# Merge utility functions
def merge2d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1]] = s
    return out

def merge3d(tensors, pad_id):
    dim1 = max([s.shape[0] for s in tensors])
    dim2 = max([s.shape[1] for s in tensors])
    dim3 = max([s.shape[2] for s in tensors])
    out = tensors[0].new(len(tensors), dim1, dim2, dim3).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :s.shape[0], :s.shape[1], :s.shape[2]] = s
    return out

def mask1d(tensors, pad_id):
    lengths = [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i, :len(s)] = 1
    return out

# Get document embeddings
def get_document_embedding(questions, image_paths, ocr_paths):
    """
    Compute document embeddings for a batch of questions, images, and OCR data.
    """
    model_name_or_path = 'pretrained_model/graphdoc'
    sentence_model_path = 'pretrained_model/sentence-bert'

    config = GraphDocConfig.from_pretrained(model_name_or_path)
    graphdoc = GraphDocForEncode.from_pretrained(model_name_or_path, config=config)
    graphdoc = graphdoc.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
    sentence_bert = AutoModel.from_pretrained(sentence_model_path)
    sentence_bert = sentence_bert.cuda().eval()

    images, bboxes, text_embeddings = [], [], []
    for question, image_path, ocr_path in zip(questions, image_paths, ocr_paths):
        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        ratio_H, ratio_W = 512 / H, 512 / W  # Calculate resizing ratios
        image = cv2.resize(image, (512, 512))

        polys, contents = read_ocr(ocr_path)
        bbox = polys2bboxes(polys)
        bbox[:, 0::2] = (bbox[:, 0::2] * ratio_W).astype('int64')
        bbox[:, 1::2] = (bbox[:, 1::2] * ratio_H).astype('int64')

        embeddings = extract_sentence_embeddings(contents, tokenizer, sentence_bert)
        question_embedding = process_question(question, tokenizer, sentence_bert)

        bbox = np.concatenate([np.array([[0, 0, 512, 512]]), bbox], axis=0)
        embeddings = np.concatenate([question_embedding[None, :], embeddings], axis=0)

        images.append(torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)))
        bboxes.append(torch.from_numpy(bbox))
        text_embeddings.append(torch.from_numpy(embeddings))

    # Compute lengths and masks before merging
    lengths = [emb.shape[0] for emb in text_embeddings]
    max_length = max(lengths)

    attention_mask_list = []
    for length in lengths:
        valid_part = torch.ones(length)
        padded_part = torch.zeros(max_length - length)
        mask = torch.cat([valid_part, padded_part])
        attention_mask_list.append(mask)

    attention_mask = torch.stack(attention_mask_list, dim=0)

    # Now merge the tensors
    images = merge3d(images, 0).cuda()
    bboxes = merge2d(bboxes, 0).cuda()
    text_embeddings = merge2d(text_embeddings, 0).cuda()

    # DO NOT overwrite attention_mask here
    attention_mask = attention_mask.cuda()  # Just move the mask to GPU

    input_data = {
        'image': images,
        'inputs_embeds': text_embeddings,
        'attention_mask': attention_mask,
        'bbox': bboxes,
        'return_dict': True
    }

    output = graphdoc(**input_data)
    return output.last_hidden_state, output.pooler_output, attention_mask


# Main function
def main():
    get_document_embedding(
        ["What is the type of organization?"],
        ["samples/ffbf0023_4.png"],
        ["samples/ffbf0023_4_easyocr.json"]
    )

if __name__ == "__main__":
    main()
