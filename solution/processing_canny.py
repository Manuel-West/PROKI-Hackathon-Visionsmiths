import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(input_image_path, output_image_path, show: bool):
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

    # Apply Canny edge detection to find edges in the image
    edges = cv2.Canny(gray_image, 50, 150)

    # Find external contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask with the same size as the grayscale image
    mask = np.zeros_like(gray_image)

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
    cv2.imwrite(output_image_path, binary_image)
    print(f"Binary image saved at {output_image_path}")

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
    moments = cv2.moments(binary_input)

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
    cv2.circle(binaryAndCenter_rgb, center_tuple, 3, (0, 255, 0), -1)  # Green circle with filled radius

    # Show the result if required
    if show:
        cv2.imshow("Binary Mask with Green Center", binaryAndCenter_rgb)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the OpenCV windows

    # Return the image with the center marked
    return binaryAndCenter_rgb


# Example usage
if __name__ == "__main__":
    # Define input and output paths for the image
    input_image_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/Rohdaten/part_4/part_4.png"  # Path to input image
    output_binary_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/output.png"  # Path to save output image
    
    # Process the image and obtain the binary result
    binary_result = process_image(input_image_path, output_binary_path)

    # Find the center of the binary mask
    _, _, center_tuple, _ = find_center(binary_result)

    # Combine the binary image with the center marked
    combine(binary_result, center_tuple, show=True)

    # Close the OpenCV windows after displaying
    cv2.waitKey(0)
    cv2.destroyAllWindows()