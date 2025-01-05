import os 
from skimage import io, segmentation, img_as_float
from skimage.color import rgb2gray
from skimage import filters

from skimage import morphology
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt





def subtract_background(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat, disk
        str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
        if light_bg:
            return black_tophat(image, str_el)
        else:
            return white_tophat(image, str_el)
        


file = '/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/data/dummy/part_1/part_1.png'

original = io.imread(file)

# Check if the image has 4 channels (RGBA)
if original.shape[-1] == 4:  # If RGBA, drop the alpha channel
    original = original[..., :3]

# Convert the image to grayscale
image = rgb2gray(original)

# Subtract background
image = subtract_background(image, radius=80, light_bg=False)

# Apply Chan-Vese segmentation
cv = segmentation.chan_vese(image, mu=0.2, lambda1=1.5, lambda2=1, tol=1e-3,
                            max_num_iter=50, dt=0.5, init_level_set="checkerboard",
                            extended_output=True)

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()




'''
# Example usage
if __name__ == "__main__":
    input_image_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/data/dummy/part_1/part_1.png"  # Input image path
    output_binary_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/output.png"       # Output path
    
    binary_result = process_image_skimage(input_image_path, output_binary_path)
    '''