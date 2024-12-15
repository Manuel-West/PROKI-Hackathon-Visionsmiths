import cv2
import matplotlib.pyplot as plt
import numpy as np

def process_image(input_image_path, output_binary_path, gamma=1.8, median_kernel_size=3):
    # Load the input image
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load the image at {input_image_path}")
    
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))

    equalized_image = clahe.apply(normalized_image)    

    blurred_image = cv2.GaussianBlur(equalized_image, (9, 9), 0)  # Lighter blur: 3x3 kernel

    gamma_corrected_image = np.array(255 * (blurred_image / 255) ** gamma, dtype=np.uint8)
    
    median_filtered_image = cv2.medianBlur(gamma_corrected_image, median_kernel_size)

    hist_median = cv2.calcHist([median_filtered_image],[0], None, [256], [0, 256])

    _, binary_image = cv2.threshold(median_filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    #binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #binary_image = cv2.bitwise_not(binary_image)


    #opening 
    kernel_open= cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

    #closing 
    #kernel_closing= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_closing)



    #TopHat
    #kernel_TopHat = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel_TopHat)




    # Visualization: Plot all stages
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original Image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Original Histogram
    axes[0, 1].plot(hist_original, color='blue')
    axes[0, 1].set_title("Original Histogram")
    axes[0, 1].set_xlim([0, 256])

    # Blurred Image
    axes[0, 2].imshow(median_filtered_image, cmap='gray')
    axes[0, 2].set_title("media Image")
    axes[0, 2].axis('off')

    # MEdain Histogram
    axes[1, 0].plot(hist_median, color='blue')
    axes[1, 0].set_title("Median Histogram")
    axes[1, 0].set_xlim([0, 256])

    # Binary Image after Otsu
    axes[1, 1].imshow(binary_image, cmap='gray')
    axes[1, 1].set_title("Binary Image (Otsu) + open + blackHAT")
    axes[1, 1].axis('off')

    # Show the figure
    plt.tight_layout()
    plt.show(block=True)

    # Save the binary image
    cv2.imwrite(output_binary_path, binary_image)

    print(f"Binary image saved at {output_binary_path}")

    return binary_image



# Example usage
if __name__ == "__main__":
    input_image_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/Rohdaten/part_1/part_1.png"  # Input image path
    output_binary_path = "/Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/output.png"       # Output path
    
    binary_result = process_image(input_image_path, output_binary_path)

    # Optionally display the binary result in a separate window
    cv2.imshow("Binary Image", binary_result)
    cv2.waitKey()
    cv2.destroyAllWindows()