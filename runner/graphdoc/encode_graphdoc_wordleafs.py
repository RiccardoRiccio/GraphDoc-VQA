def read_ocr(json_path):
    """Read OCR data and extract line-level and word-level information."""
    ocr_data = json.load(open(json_path, 'r'))
    line_polys = []
    line_contents = []
    word_polys_per_line = []
    word_contents_per_line = []

    for page in ocr_data['recognitionResults']:
        for line in page['lines']:
            line_contents.append(line['text'])
            # Line bbox
            line_bbox = [[line['boundingBox'][i], line['boundingBox'][i + 1]] 
                         for i in range(0, len(line['boundingBox']), 2)]
            line_polys.append(line_bbox)

            # Extract words and their bounding boxes
            words_in_line = []
            word_bboxes = []
            if 'words' in line:
                for w in line['words']:
                    words_in_line.append(w['text'])
                    w_bbox = [[w['boundingBox'][i], w['boundingBox'][i + 1]] 
                              for i in range(0, len(w['boundingBox']), 2)]
                    word_bboxes.append(w_bbox)
            else:
                # If words are not provided, you can split the line text
                # by spaces as a fallback (not ideal).
                # words_in_line = line['text'].split()
                # ... In this case, you won't have exact bounding boxes
                # per word, so this scenario needs handling if OCR doesn't provide words.
                pass

            word_contents_per_line.append(words_in_line)
            word_polys_per_line.append(word_bboxes)

    return line_polys, line_contents, word_polys_per_line, word_contents_per_line

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
    token_embeddings = model_output[0]  # last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (torch.sum(token_embeddings * input_mask_expanded, 1) 
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9))

def extract_line_and_word_embeddings(contents, word_contents_per_line, tokenizer, sentence_bert):
    """
    Extract embeddings for lines and also store token-level embeddings for words.
    This function returns:
    - sentence_embeddings: [num_entities, hidden_dim] line-level embeddings
    - token_embeddings_per_entity: list of tensors, each [num_tokens_in_line, hidden_dim]
    - token_word_mapping: list describing how tokens map to words
    """
    encoded_input = tokenizer(contents, padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(sentence_bert.device)
    with torch.no_grad():
        model_output = sentence_bert(**encoded_input)
    
    # Sentence-level embeddings
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()

    # Now get the token-level embeddings
    # model_output[0] is [batch_size, seq_len, hidden_size]
    all_token_embeddings = model_output[0].cpu()  # shape: [num_lines, max_seq_len, hidden_dim]
    input_ids = encoded_input['input_ids'].cpu()
    attention_mask = encoded_input['attention_mask'].cpu()

    token_embeddings_per_entity = []
    token_word_mapping = []
    
    for i, line_words in enumerate(word_contents_per_line):
        # Get the tokens for this line
        line_input_ids = input_ids[i]
        line_token_embeds = all_token_embeddings[i]
        line_attention_mask = attention_mask[i]

        # Filter out padding tokens
        valid_tokens = line_attention_mask.nonzero().squeeze()  # indices of non-padding tokens
        valid_token_embeds = line_token_embeds[valid_tokens]
        valid_input_ids = line_input_ids[valid_tokens]

        # We need to map each word in line_words to one or more tokens in valid_input_ids
        # This step is tokenizer-dependent. For simplicity, assume each word maps roughly 
        # to one token. If not, you may need more sophisticated logic:
        word_to_token_idx = []
        pointer = 0
        # Simple heuristic: re-tokenize the line's text with a basic whitespace split and 
        # try to align them word by word. A better approach is to rely on the tokenizer to 
        # provide offsets (if available) or carefully map tokens to substrings.
        line_text = contents[i]
        # Using the provided word_contents_per_line[i], we try a naive match:
        # We'll re-tokenize each word and see how it maps to the tokens.
        
        # NOTE: This part is highly dependent on tokenizer and OCR alignment.
        # A robust solution would require using special features like `tokenizer(..., return_offsets_mapping=True)` 
        # and then aligning offsets. For demonstration, we assume a direct mapping:
        
        # Attempt a naive mapping: 
        # For each word in line_words, try to find the next matching token in valid_input_ids
        # WARNING: This simplistic logic may fail depending on the tokenizer's behavior.
        # Ideally, use `tokenizer(..., return_offsets_mapping=True)` to map tokens to substrings.
        # Here, we just trust that the tokenizer does not split words too drastically.
        
        # Convert words to a lower form without punctuation if needed to match tokens:
        # (This is pseudo logic; you may need a more sophisticated approach.)
        
        # Decode tokens to strings:
        tokens_str = tokenizer.convert_ids_to_tokens(valid_input_ids)
        
        # A possible approach: For each word, find matching continuous tokens:
        current_pos = 0
        for w in line_words:
            # Find a sequence of tokens that match this word (heuristic)
            # This might be improved by using offsets.
            w_subtokens = tokenizer.tokenize(w)
            # Try to match w_subtokens starting from current_pos in tokens_str
            match_length = len(w_subtokens)
            if current_pos + match_length <= len(tokens_str) and tokens_str[current_pos:current_pos+match_length] == w_subtokens:
                word_to_token_idx.append(list(range(current_pos, current_pos+match_length)))
                current_pos += match_length
            else:
                # If we fail to match, just treat one token per word as fallback
                word_to_token_idx.append([current_pos])
                current_pos += 1

        # Now we have a mapping from words to tokens
        # We can store embeddings for each word by averaging its subword tokens:
        word_embeddings = []
        for token_group in word_to_token_idx:
            w_emb = valid_token_embeds[token_group].mean(dim=0)
            word_embeddings.append(w_emb)

        # Store token-level embeddings (subword-level)
        token_embeddings_per_entity.append(torch.stack(word_embeddings))
        token_word_mapping.append(word_to_token_idx)

    return sentence_embeddings, token_embeddings_per_entity, token_word_mapping

def get_document_embedding(questions, image_paths, ocr_paths):
    model_name_or_path = '/data2/users/rriccio/pretrained_model/graphdoc'
    sentence_model_path = '/data2/users/rriccio/pretrained_model/sentence-bert'

    config = GraphDocConfig.from_pretrained(model_name_or_path)
    graphdoc = GraphDocForEncode.from_pretrained(model_name_or_path, config=config)
    graphdoc = graphdoc.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(sentence_model_path)
    sentence_bert = AutoModel.from_pretrained(sentence_model_path)
    sentence_bert = sentence_bert.cuda().eval()

    images, bboxes, text_embeddings = [], [], []
    word_bboxes_all, word_embeddings_all = [], []

    for question, image_path, ocr_path in zip(questions, image_paths, ocr_paths):
        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        ratio_H, ratio_W = 512 / H, 512 / W
        image = cv2.resize(image, (512, 512))

        line_polys, line_contents, word_polys_per_line, word_contents_per_line = read_ocr(ocr_path)
        line_bbox = polys2bboxes(line_polys)
        # Scale line bboxes
        line_bbox[:, 0::2] = (line_bbox[:, 0::2] * ratio_W).astype('int64')
        line_bbox[:, 1::2] = (line_bbox[:, 1::2] * ratio_H).astype('int64')

        # Now handle words bounding boxes similarly
        scaled_word_bboxes_per_line = []
        for wpolys in word_polys_per_line:
            if len(wpolys) > 0:
                w_b = polys2bboxes(wpolys)
                w_b[:,0::2] = (w_b[:,0::2]*ratio_W).astype('int64')
                w_b[:,1::2] = (w_b[:,1::2]*ratio_H).astype('int64')
            else:
                w_b = np.zeros((0,4), dtype='int64')
            scaled_word_bboxes_per_line.append(w_b)

        # Extract line and word embeddings from sentence bert
        sentence_embeddings, token_embeddings_per_entity, token_word_mapping = extract_line_and_word_embeddings(
            [question] + line_contents,  # The first "entity" is the question itself
            [["question"]]+word_contents_per_line, # Just placeholder for question, if needed
            tokenizer, sentence_bert
        )

        # sentence_embeddings[0] is question, rest are lines
        # Combine question + lines bboxes: first node is the whole image region (0,0,512,512), 
        # second is question, then line entities.
        full_bbox = np.concatenate([np.array([[0,0,512,512]]), line_bbox], axis=0)
        full_embeddings = np.concatenate([sentence_embeddings[0][None,:], sentence_embeddings[1:]], axis=0)

        # Save image tensor
        images.append(torch.from_numpy(image.transpose(2,0,1).astype(np.float32)))
        bboxes.append(torch.from_numpy(full_bbox))
        text_embeddings.append(torch.from_numpy(full_embeddings))

        # For word-level embedding nodes:
        # token_embeddings_per_entity corresponds to lines (not counting the question).
        # scaled_word_bboxes_per_line matches line_contents, so indexing should be careful.
        # If we want to create a similar structure for words as nodes,
        # we can store them and use them later in graph construction.
        word_bboxes_all.append(scaled_word_bboxes_per_line)
        word_embeddings_all.append(token_embeddings_per_entity[1:]) # skip the question token embeddings

    # The rest of the code remains similar, we now have hierarchical info: 
    # text_embeddings for entity-level, and word_embeddings_all for word-level.
    # You can incorporate these into the GraphDoc model if it supports hierarchical nodes,
    # or store them for later integration.

    lengths = [emb.shape[0] for emb in text_embeddings]
    max_length = max(lengths)

    attention_mask_list = []
    for length in lengths:
        valid_part = torch.ones(length)
        padded_part = torch.zeros(max_length - length)
        mask = torch.cat([valid_part, padded_part])
        attention_mask_list.append(mask)

    attention_mask = torch.stack(attention_mask_list, dim=0)

    # Merge the tensors for graphdoc input
    images = merge3d(images, 0).cuda()
    bboxes = merge2d(bboxes, 0).cuda()
    text_embeddings = merge2d(text_embeddings, 0).cuda()
    attention_mask = attention_mask.cuda()

    input_data = {
        'image': images,
        'inputs_embeds': text_embeddings,
        'attention_mask': attention_mask,
        'bbox': bboxes,
        'return_dict': True
    }

    output = graphdoc(**input_data)

    # Now `output.last_hidden_state` and `output.pooler_output` are 
    # still the entity-level embeddings. The word-level embeddings 
    # are stored in word_embeddings_all. If you want to feed them into 
    # GraphDoc as additional nodes, you'd need to modify the GraphDoc model 
    # or the input structure to accept these additional nodes.
    return output.last_hidden_state, output.pooler_output, attention_mask, word_embeddings_all, word_bboxes_all
