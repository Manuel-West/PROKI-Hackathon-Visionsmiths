import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import jaxlie 


def process_image(input_image_path, show: bool, otsu_margin=10):
    """
    Processes the input image to remove sections identified by edges, 
    apply Otsu thresholding, and morphological operations to generate a 
    binary image.

    Parameters:
    - input_image_path (str): Path to the input image.
    - output_image_path (str): Path to save the output binary image.

    Returns:
    - binary_image (np.ndarray): Binary image after processing.
    """
    # Load the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load the image at {input_image_path}")
    
    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    #TODO
    # apply some general smotting of picture to reduce noise
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
    #cv2.imwrite(output_image_path, binary_image)
    #print(f"Binary image saved at {output_image_path}")

    return binary_image





def find_center(binary_input):
    """
    Finds the center of mass (CoM) of a binary image.

    Parameters:
    - binary_input (np.ndarray): Binary mask (single-channel, 0 or 255).

    Returns:
    - center_x, center_y (int): Coordinates of the center of mass.
    - center_tuple (tuple): Tuple of (x, y) coordinates.
    - center_array (np.ndarray): 1x2 array of the center coordinates.
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
        plt.imshow("Binary Mask with Green Center", binaryAndCenter_rgb)
        plt.title('combined mask & center')
        plt.show()

    # Return the image with the center marked
    return binaryAndCenter_rgb





def process_directory(directory_path):
    """
    Processes all subfolders in the base folder, creates binary images, finds centers,
    and plots results in a grid using Matplotlib.

    Parameters:
    - directory_path (str): Path to the base folder containing part_x subfolders.
    """
    part_folders = sorted([f for f in os.listdir(directory_path) if f.startswith("part_")])
    total_parts = len(part_folders)

    parts_per_window = 10  # Number of parts per figure

    # Process parts in chunks of 'parts_per_window'
    for window_idx in range(0, total_parts, parts_per_window):
        plt.figure(figsize=(15, 8))  # Create a new figure for each window
        plt.suptitle(f"Parts {window_idx + 1} to {min(window_idx + parts_per_window, total_parts)}")

        for idx, part_folder in enumerate(part_folders[window_idx:window_idx + parts_per_window]):
            folder_path = os.path.join(directory_path, part_folder)
            
            # Locate the first .png image in the folder
            part_image = next((f for f in os.listdir(folder_path) if f.endswith(".png")), None)
            if part_image:
                input_image_path = os.path.join(folder_path, part_image)

                # Process the image
                binary_image = process_image(input_image_path, show=False)

                # Find the center
                _, _, center_tuple, _ = find_center(binary_image)

                # Combine the binary image with the center marker
                binary_with_center = combine(binary_image, center_tuple, show=False)

                # Plot the image
                plt.subplot(2, 5, idx + 1)  # 2 rows x 5 columns = 10 subplots
                plt.imshow(binary_with_center)
                plt.title(part_folder)
                plt.axis("off")
            else:
                print(f"No PNG image found in {folder_path}.")

        plt.tight_layout()
        plt.show()  # Show the current figure without blocking execution

    print("All windows are displayed.")





def compute_SE2_transformation(center_x, center_y):
    """
    Computes the SE(2) transformation to move the origin (0, 0) to the given center.
    
    Parameters:
    - center_x (float): X-coordinate of the target center.
    - center_y (float): Y-coordinate of the target center.
    
    Returns:
    - T_obj (jaxlie.SE2): SE(2) transformation object.
    - T_Matrix: Matrix from trafo. object
    """

    # Get center array from find_center
    _, _, _, translation = find_center(center_x,center_y)

    T_obj = jaxlie.SE2.from_translation(translation)
    T_Matrix = T_obj.as_matrix()

    print(T_obj)
    print(T_Matrix)

    return T_obj, T_Matrix





# Example usage
if __name__ == "__main__":

    # Define input and output paths for the image   
    root_dir = '/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/Rohdaten/'

    process_directory(root_dir)