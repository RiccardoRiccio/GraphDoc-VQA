########################################################
# R: This code is to check the easyocr results, visualizing the bboxes in image and saving it
########################################################


import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_results(image, results):
    # Read image for visualization
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    
    # Draw boxes and text
    for result in results:
        bbox = result[0]
        text = result[1]
        
        # Get coordinates
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        tr = tuple(map(int, tr))
        br = tuple(map(int, br))
        bl = tuple(map(int, bl))
        
        # Draw box
        plt.plot([tl[0], tr[0], br[0], bl[0], tl[0]], 
                 [tl[1], tr[1], br[1], bl[1], tl[1]], 
                 'r-', linewidth=2)
        
        # Add text
        plt.text(tl[0], tl[1], text, 
                bbox=dict(facecolor='white', alpha=0.7),
                fontsize=12, color='blue')
    
    plt.axis('off')
    
    # Save the figure instead of showing it
    output_path = image.replace('.png', '_ocr_visualization.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"\nVisualization saved to: {output_path}")

def main():
    try:
        # Initialize the OCR reader
        reader = easyocr.Reader(['en'])
        
        # Path to your image (adjust this path as needed)
        image_path = "/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4.png"
        
        # Read the image and get results with paragraph grouping
        results = reader.readtext(
            image_path,
            paragraph=True,
            width_ths=0.7,  # Width threshold for grouping
            add_margin=0.1  # Margin to add around the paragraph
        )
        
        # Print detected text and bounding boxes
        print("OCR Results (Paragraph Level):")
        print("-" * 50)
        for result in results:
            bbox = result[0]      # bounding box
            text = result[1]      # detected text
            
            print(f"Paragraph Text: {text}")
            print(f"Bounding Box Coordinates: {bbox}")
            print("-" * 50)
        
        # Visualize results
        visualize_results(image_path, results)
        
        # Optional: Save results to a text file
        with open('ocr_results.txt', 'w', encoding='utf-8') as f:
            for result in results:
                bbox = result[0]
                text = result[1]
                f.write(f"Paragraph Text: {text}\n")
                f.write(f"Bounding Box: {bbox}\n")
                f.write("-" * 50 + "\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTrying to fix the error...")
        print("Please run the following commands in your terminal:")
        print("pip uninstall easyocr")
        print("pip install easyocr==1.6.2")
        print("\nThen run this script again.")

if __name__ == "__main__":
    main()