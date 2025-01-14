import os
import numpy as np

def main():
    # Path to your imdb_train.npy file (adjust the path if needed)
    imdb_file = "/SSD2/Datasets/DocVQA/Task1/pythia_data/imdb/docvqa/new_imdb_train.npy"
    
    # Load the numpy file; allow_pickle is required as the file contains a complex structure.
    data = np.load(imdb_file, allow_pickle=True)
    
    # The first element is a header, and the rest are records.
    header = data[0]
    records = data[1:]
    
    print("Header:")
    print(header)
    print(f"Total number of records: {len(records)}")
    
    # How many records you want to inspect:
    num_to_inspect = 5

    for i, record in enumerate(records[:num_to_inspect]):
        print("\n" + "="*40)
        print(f"Record {i+1}")
        
        # Print available keys
        print("Keys available:", record.keys())
        
        # Get OCR boxes using the key "ocr_normalized_boxes"
        if "ocr_normalized_boxes" in record:
            boxes = record["ocr_normalized_boxes"]
            boxes = np.array(boxes)
            print(f"Boxes shape: {boxes.shape}")
            
            # Compute overall statistics of the bounding boxes:
            print("Min value in boxes:", boxes.min())
            print("Max value in boxes:", boxes.max())
            print("Mean value in boxes:", boxes.mean())
            
            # Optionally, print the first few boxes
            print("First 5 boxes:")
            print(boxes[:5])
        else:
            print("No 'ocr_normalized_boxes' key found in this record.")

        # Optionally, also print OCR tokens to see context
        if "ocr_tokens" in record:
            tokens = record["ocr_tokens"]
            print("Number of OCR tokens:", len(tokens))
            print("First 10 OCR tokens:", tokens[:10])
        else:
            print("No 'ocr_tokens' key found in this record.")

if __name__ == "__main__":
    main()
