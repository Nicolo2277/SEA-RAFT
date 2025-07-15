import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_segmentation_mask(image_path):
    # Load the image and convert to numpy array
    mask = Image.open(image_path)
    mask_np = np.array(mask)

    # Define colors: background (black), class 1 (red), class 2 (green)
    colors = np.array([
        [0, 0, 0], 
        [255, 0, 0],      # background: black
        [255, 255, 0],     # class 1: yellow     # class 2: red
    ], dtype=np.uint8)

    # Create an RGB image using the color map
    color_mask = colors[mask_np]

    # Plot the colored mask
    plt.figure(figsize=(6, 6))
    plt.imshow(color_mask)
    plt.axis('off')
    plt.title('Segmentation Mask')
    plt.show()

# Example usage
#IMAGE_PATH = "plot_gt_pred/pred_mask.png"
IMAGE_PATH = "plot_gt_pred/ground_truth_mask.png"

plot_segmentation_mask(IMAGE_PATH)
