########################################################
# R: This code is to create the easyocr json file from the image
########################################################

import easyocr
import cv2
import json
from pathlib import Path

def easyocr_to_json(image_path, word_results, paragraph_results):
    # Read image to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    json_data = {
        "status": "Succeeded",
        "recognitionResults": [
            {
                "page": 1,
                "clockwiseOrientation": 0,
                "width": width,
                "height": height,
                "unit": "pixel",
                "lines": [],  # for paragraphs/regions
                # "words": []   # for individual words
            }
        ]
    }

    # Add paragraph/region level results
    for result in paragraph_results:
        bbox, text = result
        flat_bbox = [int(coord) for point in bbox for coord in point]
        
        line = {
            "boundingBox": flat_bbox,
            "text": text
        }
        json_data["recognitionResults"][0]["lines"].append(line)

    # # Add word level results
    # for result in word_results:
    #     bbox, text, conf = result
    #     flat_bbox = [int(coord) for point in bbox for coord in point]
        
    #     word = {
    #         "boundingBox": flat_bbox,
    #         "text": text,
    #         "confidence": float(conf)
    #     }
    #     json_data["recognitionResults"][0]["words"].append(word)

    return json_data

def main():
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'])
    
    # Path to your image
    image_path = "/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4.png"
    
    # Get word-level results with smaller width_ths
    # word_results = reader.readtext(
    #     image_path,
    #     paragraph=False,
    #     width_ths=0.1  # Smaller value to prevent word merging
    # )
    
    # Get paragraph-level results with default parameters
    paragraph_results = reader.readtext(
        image_path,
        paragraph=True,
        width_ths=0.7,
        add_margin=0.1
    )
    
    # Convert to JSON format
    json_data = easyocr_to_json(image_path, word_results, paragraph_results)
    
    # Save JSON to file
    output_json_path = str(Path(image_path).with_suffix('')) + '_easyocr.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"OCR results saved to: {output_json_path}")

if __name__ == "__main__":
    main()