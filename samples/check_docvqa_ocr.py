import cv2
import json
import matplotlib.pyplot as plt

# Path to the JSON and image files
json_file_path = "/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4.json"  # Path to your JSON file
image_file_path = "/home/rriccio/Desktop/GraphDoc/samples/ffbf0023_4.png"  # Path to your image file

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Load the image
img = cv2.imread(image_file_path)

# Convert to RGB for visualization (OpenCV uses BGR by default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Iterate through the lines and draw bounding boxes
for result in data["recognitionResults"]:
    for line in result["lines"]:
        # Extract bounding box coordinates
        bbox = line["boundingBox"]
        points = [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]  # (x, y) pairs

        # Draw the rectangle around the line
        pt1 = points[0]
        pt2 = points[2]
        img_rgb = cv2.rectangle(img_rgb, pt1, pt2, (255, 0, 0), 2)  # Blue box for line

# Show the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')  # Hide axes
plt.show()

# Save the image with bounding boxes
output_image_path = '/home/rriccio/Desktop/GraphDoc/samples/docvqaline_bboxes.png'  # Output path for the image
cv2.imwrite(output_image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
