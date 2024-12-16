import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
from spellchecker import SpellChecker


def visualize_results(image, results, level="paragraph"):
    # Read image for visualization
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    plt.figure(figsize=(20, 20))
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
                 'r-' if level == "paragraph" else 'b-', linewidth=2)
    
    plt.axis('off')
    
    # Save the figure instead of showing it
    output_path = image.replace('.png', f'_ocr_visualization_{level}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"\n{level.capitalize()} level visualization saved to: {output_path}")


def save_results_with_corrections(results, spell_checker, level="paragraph", output_file="ocr_results_with_corrections.txt"):
    # Open the file to save results
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{level.capitalize()} Level OCR Results:\n")
        f.write("-" * 50 + "\n")
        for result in results:
            text = result[1]
            corrected_text = spell_checker.correction(text)
            bbox = result[0]
            f.write(f"Original Text: {text}\n")
            f.write(f"Corrected Text: {corrected_text}\n")
            f.write(f"Bounding Box: {bbox}\n")
            f.write("-" * 50 + "\n")
        f.write("\n")
    
    print(f"\n{level.capitalize()} level results with corrections saved to: {output_file}")


def main():
    try:
        # Initialize the OCR reader and spell checker
        reader = easyocr.Reader(['en'])
        spell_checker = SpellChecker()
        
        # Path to your image (adjust this path as needed)
        image_path = "/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4.png"
        
        # Read the image and get results with paragraph grouping
        results_paragraph = reader.readtext(
            image_path,
            paragraph=True,
            # width_ths=0.7,  # Width threshold for grouping
            # add_margin=0.1  # Margin to add around the paragraph
        )
        
        # Read the image and get results without paragraph grouping (word level)
        results_word = reader.readtext(
            image_path,
            paragraph=False,  # Disable paragraph grouping
            width_ths=0.07,    # Stricter grouping for single words
            add_margin=0,     # No additional margin
            x_ths=0.1,        # Limit horizontal box merging
            y_ths=0.2,        # Limit vertical box merging
            detail=1          # Get bounding boxes and confidence scores
        )

        # Save results with spelling corrections
        # save_results_with_corrections(results_paragraph, spell_checker, level="paragraph")
        # save_results_with_corrections(results_word, spell_checker, level="word")
        
        # Visualize paragraph level results
        visualize_results(image_path, results_paragraph, level="paragraph")
        
        # Visualize word level results
        # visualize_results(image_path, results_word, level="word")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTrying to fix the error...")
        print("Please run the following commands in your terminal:")
        print("pip uninstall easyocr")
        print("pip install easyocr==1.6.2")
        print("\nThen run this script again.")


if __name__ == "__main__":
    main()
