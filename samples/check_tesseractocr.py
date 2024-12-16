import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt

# Path to Tesseract executable (adjust this for your setup)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Example for Linux

def visualize_results(image, results):
    # Read image for visualization
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    
    # Draw boxes and text
    for bbox, text in results:
        (x, y, w, h) = bbox
        
        # Draw box
        plt.plot([x, x + w, x + w, x, x], 
                 [y, y, y + h, y + h, y], 
                 'r-', linewidth=2)
        
        # Add text
        plt.text(x, y - 10, text, 
                 bbox=dict(facecolor='white', alpha=0.7),
                 fontsize=12, color='blue')
    
    plt.axis('off')
    
    # Save the figure instead of showing it
    output_path = image.replace('.png', '_ocr_visualization_tesseract.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"\nVisualization saved to: {output_path}")

def main():
    try:
        # Path to your image (adjust this path as needed)
        image_path = "/home/rriccio/Desktop/GraphDoc/samples/image_before2.png"
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Perform OCR with Tesseract (PSM 6 for word level)
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 11"  # PSM 6 assumes a single uniform block of text for word-level analysis
        )
        
        # Parse results
        results = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Confidence > 0
                x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                text = data['text'][i]
                results.append(((x, y, w, h), text))
        
        # Print detected text and bounding boxes
        print("OCR Results (Word Level):")
        print("-" * 50)
        for bbox, text in results:
            print(f"Word Text: {text}")
            print(f"Bounding Box Coordinates: {bbox}")
            print("-" * 50)
        
        # Visualize results
        visualize_results(image_path, results)
        
        # Optional: Save results to a text file
        with open('ocr_results_tesseract_word_level.txt', 'w', encoding='utf-8') as f:
            for bbox, text in results:
                f.write(f"Word Text: {text}\n")
                f.write(f"Bounding Box: {bbox}\n")
                f.write("-" * 50 + "\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
