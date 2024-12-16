import cv2
import json
import torch
import numpy as np
import sys
sys.path.append('./')

from layoutlmft.models.graphdoc.configuration_graphdoc import GraphDocConfig
from layoutlmft.models.graphdoc.modeling_graphdoc import GraphDocForEncode
from transformers import AutoModel, AutoTokenizer

##############
# description of changes:
#   SAme as encode_document.py but here i save the embedding and attention mask as .pt file
##############


# R: new read_ocr function for getting word level information (from docVQA dataset)
# def read_ocr(json_path):
#     """Read OCR data and extract word-level information."""
#     ocr_data = json.load(open(json_path, 'r'))  # Load JSON
#     polys = []  # Store bounding boxes
#     contents = []  # Store word texts

#     # Navigate the JSON hierarchy
#     for page in ocr_data['recognitionResults']:
#         for line in page['lines']:
#             for word in line['words']:
#                 contents.append(word['text'])  # Extract text
#                 # Convert [x1, y1, x2, y2, x3, y3, x4, y4] to [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#                 bbox = [[word['boundingBox'][i], word['boundingBox'][i + 1]] for i in range(0, len(word['boundingBox']), 2)]
#                 polys.append(bbox)  # Append bounding box

#     return polys, contents

# R: new read_ocr function for getting region level information (from easyocr json file like GraphDoc paper)
def read_ocr(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    # Extract lines (paragraphs) from the JSON
    lines = ocr_data['recognitionResults'][0]['lines']
    
    polys = []
    contents = []
    for line in lines:
        bbox = line['boundingBox']
        text = line['text']
        
        # Convert flat bbox to polygon format
        poly = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[4], bbox[5]],
            [bbox[6], bbox[7]]
        ]
        
        polys.append(poly)
        contents.append(text)
    
    return polys, contents

# R: new process_question function
def process_question(question, tokenizer, sentence_bert):
    encoded_input = tokenizer([question], padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    question_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
    return question_embedding[0]

# previous read_ocr function
# def read_ocr(json_path):
#     ocr_info = json.load(open(json_path, 'r'))
#     polys = []
#     contents = []
#     for info in ocr_info:
#         contents.append(info['label'])
#         polys.append(info['points'])
#     return polys, contents


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
def get_document_embedding(question, image_path, ocr_path):

    model_name_or_path = 'pretrained_model/graphdoc'
    sentence_model_path = 'pretrained_model/sentence-bert'
    # image_path = 'samples/ffbf0023_4.png'
    # ocr_path = 'samples/ffbf0023_4.json'

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
    # global_bbox = np.array([0, 0, 512,512]).astype('int64')
    # bboxes = np.concatenate([global_bbox[None, :], bboxes], axis=0)
    # global_embed = np.zeros_like(sentence_embeddings[0])
    # sentence_embeddings = np.concatenate([global_embed[None, :], sentence_embeddings], axis=0)

    # R: append question node instead of global node
    # append question node instead of global node
    # question = "What is the type of organization?"  # Your question here
    question_embedding = process_question(question, tokenizer, sentence_bert)
    global_bbox = np.array([0, 0, 512, 512]).astype('int64')
    bboxes = np.concatenate([global_bbox[None, :], bboxes], axis=0)
    sentence_embeddings = np.concatenate([question_embedding[None, :], sentence_embeddings], axis=0)


    input_images = merge3d([torch.from_numpy(image.transpose(2,0,1).astype(np.float32))], 0).cuda()
    input_embeds = merge2d([torch.from_numpy(sentence_embeddings)], 0).cuda()
    attention_mask = mask1d([torch.from_numpy(sentence_embeddings)], 0).cuda()
    input_bboxes = merge2d([torch.from_numpy(bboxes)], 0).cuda()
    input_data=dict(image=input_images, inputs_embeds=input_embeds, attention_mask=attention_mask, bbox=input_bboxes, return_dict=True)

    output = graphdoc(**input_data)

    # print(output)
    
     # Attention mask
    attention_mask = input_data['attention_mask']

     # Print image dimensions
    print("\nInput Dimensions:")
    print(f"Original Image (H, W): {H}, {W}")
    print(f"Resized Image (H, W): {input_H}, {input_W}")
    
    # After preparing OCR data
    print("\nOCR Data:")
    print(f"Number of text boxes: {len(polys)}")
    print(f"Number of text contents: {len(contents)}")
    
    # After preparing embeddings
    print("\nEmbedding Dimensions:")
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")
    print(f"Question embedding shape: {question_embedding.shape}")
    print(f"Bounding boxes shape: {bboxes.shape}")
    
    # After preparing input tensors
    print("\nInput Tensor Dimensions:")
    print(f"Input images shape: {input_images.shape}")
    print(f"Input embeds shape: {input_embeds.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Input bboxes shape: {input_bboxes.shape}")
    
    # After model output
    print("\nOutput Dimensions:")
    print(f"Last hidden state shape: {output.last_hidden_state.shape}")
    print(f"Pooler output shape: {output.pooler_output.shape}")
    
    return output.last_hidden_state, output.pooler_output, input_data['attention_mask']
    
def main():
    # Example paths
    question = "What is the type of organization?"
    image_path = "samples/ffbf0023_4.png"
    ocr_path = "samples/ffbf0023_4_easyocr.json"
    
    # Get embeddings
    last_hidden_state, pooler_output, attention_mask = get_document_embedding(question, image_path, ocr_path)

    # Create output filename based on input image name
    output_name = Path(image_path).stem + "_embeddings.pt"
    save_path = str(Path(image_path).parent / output_name)
    
    # Save embeddings and attention mask
    torch.save({
        'last_hidden_state': last_hidden_state,
        'pooler_output': pooler_output,
        'attention_mask': attention_mask
    }, save_path)
    
    print(f"Saved embeddings and attention mask to {save_path}")
    
    # Print shapes for verification
    print("\nEmbedding shapes:")
    print(f"Last hidden state: {last_hidden_state.shape}")
    print(f"Pooler output: {pooler_output.shape}")
    print(f"Attention mask: {attention_mask.shape}")

if __name__ == "__main__":
    from pathlib import Path
    main()