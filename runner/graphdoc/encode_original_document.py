import cv2
import json
import torch
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append('./')

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from transformers import AutoModel, AutoTokenizer


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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def extract_sentence_embeddings(contents, tokenizer, sentence_bert):
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input= encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return sentence_embeddings

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
    lengths= [len(s) for s in tensors]
    out = tensors[0].new(len(tensors), max(lengths)).fill_(pad_id)
    for i, s in enumerate(tensors):
        out[i,:len(s)] = 1
    return out

def get_document_embedding(image_path, ocr_path, save_dir):
    model_name_or_path = 'pretrained_model/graphdoc'
    sentence_model_path = 'pretrained_model/sentence-bert'
    

    # init model
    config = GraphDocConfig.from_pretrained(model_name_or_path)
    graphdoc = GraphDocForEncode.from_pretrained(model_name_or_path, config=config)
    graphdoc = graphdoc.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
    sentence_bert = AutoModel.from_pretrained(sentence_model_path)
    sentence_bert = sentence_bert.cuda().eval()

    # prepare input data
    input_H = 512; input_W = 512
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    ratio_H = input_H / H; ratio_W = input_W / W
    image = cv2.resize(image, dsize=(input_W, input_H))
    polys, contents = read_ocr(ocr_path)
    bboxes = polys2bboxes(polys)
    bboxes[:, 0::2] = bboxes[:, 0::2] * ratio_W
    bboxes[:, 1::2] = bboxes[:, 1::2] * ratio_H
    sentence_embeddings = extract_sentence_embeddings(contents, tokenizer, sentence_bert)

    # append global node
    global_bbox = np.array([0, 0, 512,512]).astype('int64')
    bboxes = np.concatenate([global_bbox[None, :], bboxes], axis=0)
    global_embed = np.zeros_like(sentence_embeddings[0])
    sentence_embeddings = np.concatenate([global_embed[None, :], sentence_embeddings], axis=0)

    input_images = merge3d([torch.from_numpy(image.transpose(2,0,1).astype(np.float32))], 0).cuda()
    input_embeds = merge2d([torch.from_numpy(sentence_embeddings)], 0).cuda()
    attention_mask = mask1d([torch.from_numpy(sentence_embeddings)], 0).cuda()
    input_bboxes = merge2d([torch.from_numpy(bboxes)], 0).cuda()
    input_data=dict(image=input_images, inputs_embeds=input_embeds, attention_mask=attention_mask, bbox=input_bboxes, return_dict=True)

    output = graphdoc(**input_data)
    print(output)
        # Extract necessary tensors
    hidden_state = output['last_hidden_state']
    pooler_output = output['pooler_output']

    # Ensure tensors retain the batch dimension
    hidden_state = hidden_state.unsqueeze(0) if hidden_state.ndim == 2 else hidden_state
    pooler_output = pooler_output.unsqueeze(0) if pooler_output.ndim == 1 else pooler_output
    attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

    # Save the embeddings to a .pt file
    save_path = os.path.join(save_dir, f"{Path(image_path).stem}.pt")
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    torch.save({
        "last_hidden_state": hidden_state.cpu(),
        "pooler_output": pooler_output.cpu(),
        "attention_mask": attention_mask.cpu(),
        "image_path": image_path,
        "ocr_path": ocr_path
    }, save_path)
    print(f"Embeddings saved to {save_path}")



# Main function
def main():
    save_dir = "/home/rriccio/Desktop/GraphDoc/ORIGINAL_EMBEDDINGS"
    get_document_embedding(
        "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_images/ffng0227_13.png",
        "/home/rriccio/Desktop/GraphDoc/sp-docvqa/spdocvqa_ocr/ffng0227_13.json",
        save_dir
    )

if __name__ == "__main__":
    main()