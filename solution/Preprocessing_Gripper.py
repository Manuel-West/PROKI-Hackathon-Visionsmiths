import cv2
import os
from pathlib import Path

def get_binary_image(path):
    """
    Convert an image to a binary image using Otsu's thresholding.

    This function reads an image from the specified path, converts it to grayscale,
    and then applies Otsu's thresholding to create a binary image.

    Parameters:
    path (str): The file path to the image.

    Returns:
    numpy.ndarray: The binary image if the file exists and can be read; otherwise, None.

    """
    # Read RGB image
    img = cv2.imread(path)

    # Convert image to grayscale image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image using Otsu's thresholding (Histogram based)
    ret,thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def find_center(binary_image, show = False):
    """
    Calculate the center of mass of a binary image.

    This function computes the moments of the binary image to determine the center of mass.
    Optionally, it can display the image with the center highlighted.

    Parameters:
    binary_image (numpy.ndarray): The binary image for which the center of mass is to be calculated.
    show (bool, optional): If True, displays the image with the center highlighted. Default is False.

    Returns:
    tuple: A tuple containing the x and y coordinates of the center of mass (cX, cY).

    Raises:
    ValueError: If the binary image is empty or if the moments cannot be calculated.
    """
    # calculate moments of binary image
    M = cv2.moments(binary_image)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # mainly for debugging
    if show:
        image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        # put text and highlight the center
        # image center
        cv2.circle(image, (0, 0), 5, (0, 0, 255), -1)
        # gripper center
        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)

        # display the image
        cv2.imshow("Image", image)
        print("Press any key to continue...")
        cv2.waitKey(0)
    return cX, cY

def preprocessing_gripper(image_path, show = False):
    """
    Preprocess an image to find the center of mass of a gripper.

    This function performs the following steps:
    1. Converts the image to a binary image using Otsu's thresholding.
    2. Calculates the center of mass of the binary image.

    Parameters:
    image_path (str): The file path to the image.
    show (bool, optional): If True, displays intermediate images. Default is False.

    Returns:
    tuple: A tuple containing the x and y coordinates of the center of mass (cX, cY).
    """
    binary_image = get_binary_image(image_path )
    cX, cY = find_center(binary_image, show=show)
    return cX, cY

if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = Path(script_dir).parent

    relative_paths = ['data/Gripper_test_1.png',
                      'data/Rohdaten/part_1/1.png', 'data/Rohdaten/part_1/4.png',
                      'data/Rohdaten/part_4/2.png', 'data/Rohdaten/part_6/5.png',
                      'data/Rohdaten/part_8/3.png',
                      ]
    for relative_path in relative_paths:
        # Construct the relative path to the image
        image_path = os.path.join(root_dir, relative_path)

        cX, cY = preprocessing_gripper(image_path, show=True)
        print(f"cX: {cX}, cY: {cY}")