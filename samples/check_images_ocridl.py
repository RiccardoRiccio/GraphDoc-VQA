from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib for headless or GUI environments
try:
    matplotlib.use('TkAgg')  # For GUI display
except ImportError:
    matplotlib.use('Agg')  # For headless environments

# Image path
image_path = "/home/rriccio/Desktop/GraphDoc/pretrain_dataset/ffbb0002.tif"
output_image_path = "/home/rriccio/Desktop/GraphDoc/output_image.png"

print(f"Opening image: {image_path}")

# Load image and convert to PNG
try:
    # Open the .tif image
    img = Image.open(image_path)
    print(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")

    # Convert and save the image to PNG format
    img.save(output_image_path, format="PNG")
    print(f"Image converted and saved to: {output_image_path}")

    # Display the converted image
    img_png = Image.open(output_image_path)
    plt.imshow(img_png)
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Error loading, converting, or displaying image: {e}")
