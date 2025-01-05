import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def process_image(input_image_path, output_path, show: bool, inverted_binary: bool, otsu_margin=10, save=False):
    """
    Processes an input image to create a binary representation by identifying 
    and masking sections using Canny edge detection, Otsu thresholding, and 
    morphological operations.

    Parameters:
    - input_image_path (str): Path to the input image file.
    - output_path (str): Path to save the resulting binary image and its name.
    - show (bool): Whether to display intermediate processing steps using Matplotlib.
    - inverted_binary (bool): Whether to generate and return an inverted binary image.
    - otsu_margin (int): Margin factor for computing thresholds for edge detection.
    - save (bool): Whether to save the processed image or not.

    Returns:
    - binary_image (np.ndarray): The processed binary image.
    - binary_inverted (np.ndarray): The inverted binary image, if requested; otherwise `None`.
    """
    
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load the image at {input_image_path}")
    
    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply some general smoothing of picture to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)  # Lighter blur: 3x3 kernel

    # get OTSU thresholds (scalar) for canny
    otsu_thresh, _ = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = max(0, int(otsu_thresh * (1 - otsu_margin)))
    upper = min(255, int(otsu_thresh * (1 + otsu_margin)))

    # Apply Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred_image, lower, upper)
    
    #closing the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  # Small kernel to close gaps
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find external contours of the edges
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask with the same size as the grayscale image
    mask = np.zeros_like(blurred_image)

    # Draw the detected contours on the mask (filled in white)
    _ = cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Copy the grayscale image and set the masked regions to black
    result_image = gray_image.copy()
    result_image[mask == 255] = 0  # Set the masked areas to black

    # Apply Otsu's thresholding to convert the image to binary
    _, binary_image = cv2.threshold(result_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    if show: 
        # Display the results using matplotlib
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Canny edges
        plt.subplot(1, 4, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edges')
        plt.axis('off')

        # Zones identified by Canny
        plt.subplot(1, 4, 3)
        plt.imshow(result_image, cmap='gray')
        plt.title('Canny Zones')
        plt.axis('off')

        # Final binary image
        plt.subplot(1, 4, 4)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Save the final binary image
    if save:
        cv2.imwrite(output_path, binary_image)
    
    binary_inverted = None
    
    if inverted_binary:
        binary_inverted = cv2.bitwise_not(binary_image)
    
    # get dimensions of the binary for postprocessing
    height, width = binary_image.shape
    shape = (width, height)

    return binary_image, binary_inverted, shape




def find_center(binary_input):
    """
    Calculates the center of mass (CoM) for a given binary image.

    Parameters:
    - binary_input (np.ndarray): Binary mask image (single-channel, 0 or 255).

    Returns:
    - center_x (int): X-coordinate of the center of mass.
    - center_y (int): Y-coordinate of the center of mass.
    - center_tuple (tuple): Tuple containing (x, y) coordinates of the center.
    - center_array (np.ndarray): Array representation of the center coordinates.
    """
       
    # Calculate moments of the binary image
    moments = cv2.moments(cv2.bitwise_not(binary_input))

    # Initialize the center tuple
    center_tuple = ()
    
    # Check if moments are valid (avoid division by zero)
    if moments["m00"] != 0:
        # Extract center of mass (CoM) coordinates
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
 
        # Update the center tuple
        center_tuple = (center_x, center_y)

        # Convert the center tuple to a NumPy array for further processing
        center_array = np.array(center_tuple)

    else:
        # If no valid center found, return None
        center_x, center_y = None, None
        center_array = np.array([None, None])

    # Return center coordinates and the array containing center coordinates
    return center_x, center_y, center_tuple, center_array





def combine(binary, center_tuple: tuple, show: bool):
    """
    Combines the binary image with a visual marker at the center.

    Parameters:
    - binary (np.ndarray): The binary image.
    - center_tuple (tuple): The coordinates of the center.
    - show (bool): Whether to display the image with the center marked.

    Returns:
    - binaryAndCenter_rgb (np.ndarray): The binary image with center marked.
    """
    
    # Make a copy of the binary image to overlay the center
    binaryAndCenter = binary.copy()

    # Convert the binary image to RGB for displaying the center in color
    binaryAndCenter_rgb = cv2.cvtColor(binaryAndCenter, cv2.COLOR_GRAY2BGR)

    # Draw a green circle at the center
    _ = cv2.circle(binaryAndCenter_rgb, center_tuple, 3, (0, 255, 0), -1)  # Green circle with filled radius

    # Show the result if required
    if show:
            plt.imshow(binaryAndCenter_rgb)  # Pass the image directly
            plt.title('Binary Mask with Green Center')  # Set the title
            plt.axis('off')  # Hide the axes
            plt.show()

    # Return the image with the center marked
    return binaryAndCenter_rgb





# Example usage
if __name__ == "__main__":

    #usage: python3 processing_canny.py  path/to/picture.png  output/path/to/solution.png 

    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    binary_image, binary_image_invert, shape = process_image(inputPath, outputPath, show= False, inverted_binary= False)
    
    _, _, centerTuple, _ = find_center(binary_image)
    combined_image = combine(binary_image, centerTuple, show= True)
    
    print("Shape of the image (x,y):", shape)
    
