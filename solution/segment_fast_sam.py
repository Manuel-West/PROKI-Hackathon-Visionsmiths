from ultralytics import FastSAM
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageSegmenter:
    def __init__(self, model_path="FastSAM-s.pt"):
        """
        Initialize the ImageSegmenter with a FastSAM model.

        Args:
            model_path (str): Path to the FastSAM model weights
        """
        self.model = FastSAM(model_path)

    def segment_image(self, image_path, device="cpu", conf=0.4, iou=0.9, imgsz=1024):
        """
        Segment an image and return the results.

        Args:
            image_path (str): Path to the input image
            device (str): Device to run inference on ('cpu' or 'cuda')
            conf (float): Confidence threshold
            iou (float): IoU threshold
            imgsz (int): Input image size

        Returns:
            tuple: (original image, segmentation results)
        """
        # Load and process the image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = self.model(
            image_path,
            device=device,
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou
        )

        return original_image, results[0]

    def visualize_all_segments(self, original_image, results, save_path=None):
        """
        Visualize all segmentation masks with different colors.

        Args:
            original_image (np.array): Original input image
            results: Segmentation results from FastSAM
            save_path (str, optional): Path to save the visualization
        """
        # Create a figure with subplots - original, all segments, and individual masks
        masks = results.masks.data.cpu().numpy()
        n_masks = len(masks)

        # Calculate number of rows needed for subplot grid
        n_cols = 3  # We'll show 3 images per row
        n_rows = 2 + (n_masks - 1) // n_cols  # Add 2 rows for original and all segments

        fig = plt.figure(figsize=(15, 5 * n_rows))

        # Plot original image
        ax1 = plt.subplot(n_rows, n_cols, 1)
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Plot all segments together
        ax2 = plt.subplot(n_rows, n_cols, 2)
        segmented_image = original_image.copy()

        # Generate random colors for each mask
        colors = np.random.rand(n_masks, 3)

        # Create combined visualization
        for idx, mask in enumerate(masks):
            bool_mask = mask.astype(bool)
            color_mask = np.zeros_like(original_image, dtype=np.float32)
            color_mask[bool_mask] = colors[idx]
            segmented_image = cv2.addWeighted(
                segmented_image, 1,
                (color_mask * 255).astype(np.uint8), 0.5,
                0
            )

        ax2.imshow(segmented_image)
        ax2.set_title(f'All Segments ({n_masks} masks)')
        ax2.axis('off')

        # Plot individual masks
        for idx, mask in enumerate(masks):
            ax = plt.subplot(n_rows, n_cols, idx + 3)  # +3 because first two spots are taken

            # Create visualization for individual mask
            mask_viz = original_image.copy()
            bool_mask = mask.astype(bool)
            color_mask = np.zeros_like(original_image, dtype=np.float32)
            color_mask[bool_mask] = colors[idx]
            mask_viz = cv2.addWeighted(
                mask_viz, 1,
                (color_mask * 255).astype(np.uint8), 0.5,
                0
            )

            ax.imshow(mask_viz)

            # Calculate mask statistics
            mask_area = np.sum(bool_mask)
            coverage_percentage = (mask_area / bool_mask.size) * 100
            ax.set_title(f'Mask {idx + 1}\n{coverage_percentage:.1f}% coverage')
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"Visualization saved to {save_path}")

        plt.show()

        # Print overall statistics
        print("\nSegmentation Statistics:")
        print(f"Total number of segments: {n_masks}")
        for idx, mask in enumerate(masks):
            mask_area = np.sum(mask.astype(bool))
            coverage = (mask_area / mask.size) * 100
            print(f"Mask {idx + 1}: {mask_area} pixels ({coverage:.2f}% coverage)")

    def calculate_background(self, original_image, results, min_threshold=0.001, max_threshold=0.1):
        """
        Calculate background mask and process the image.

        Args:
            original_image (np.array): Original input image
            results: Segmentation results from FastSAM
            min_threshold (float): Minimum size threshold as a fraction of image size (0.0 to 1.0)
            max_threshold (float): Maximum size threshold as a fraction of image size (0.0 to 1.0)

        Returns:
            np.array: Processed and binarized image as numpy array
        """
        masks = results.masks.data.cpu().numpy()

        # Filter masks based on size thresholds
        total_pixels = masks[0].size
        min_pixels = total_pixels * min_threshold
        max_pixels = total_pixels * max_threshold

        # Create combined mask of segments within threshold range
        combined_mask = np.zeros(masks[0].shape, dtype=bool)
        filtered_masks = []
        too_small_masks = []
        too_large_masks = []

        for mask in masks:
            bool_mask = mask.astype(bool)
            mask_size = np.sum(bool_mask)

            if min_pixels <= mask_size < max_pixels:
                combined_mask = combined_mask | bool_mask
                filtered_masks.append(mask)
            elif mask_size < min_pixels:
                too_small_masks.append(mask)
            else:
                too_large_masks.append(mask)

        # Create a copy of the original image as numpy array
        processed_image = original_image.copy()

        # Set pixels to zero where the combined mask is True
        processed_image[combined_mask] = 0

        # Binarize the resulting image
        if len(processed_image.shape) > 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        binary_output = (processed_image > 0).astype(np.uint8)

        return binary_output, ~binary_output

    def visualize_segmented_part(self, original_image, part_mask, combined_mask, save_path=None):
        """
        Visualize the part mask and combined mask overlaid on the original image.

        Args:
            original_image (np.array): Original input image
            part_mask (np.array): Binary mask of background/part regions
            combined_mask (np.array): Binary mask of combined segments
            save_path (str, optional): Path to save visualization. If None, displays only.
        """

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 10))

        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot part mask
        axes[1].imshow(part_mask, cmap='gray')
        axes[1].set_title('Part Mask')
        axes[1].axis('off')

        # Plot combined mask
        axes[2].imshow(combined_mask, cmap='gray')
        axes[2].set_title('Combined Segments Mask')
        axes[2].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


    def visualize_background(self, original_image, results, min_threshold=0.02, max_threshold=0.1, save_path=None):
        """
        Show segments within the specified size range.

        Args:
            original_image (np.array): Original input image
            results: Segmentation results from FastSAM
            min_threshold (float): Minimum size threshold as a fraction of image size (0.0 to 1.0)
            max_threshold (float): Maximum size threshold as a fraction of image size (0.0 to 1.0)
            save_path (str, optional): Path to save the visualization
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot original image
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Get masks from results
        masks = results.masks.data.cpu().numpy()

        # Filter masks based on size thresholds
        total_pixels = masks[0].size
        min_pixels = total_pixels * min_threshold
        max_pixels = total_pixels * max_threshold

        # Create combined mask of segments within threshold range
        combined_mask = np.zeros(masks[0].shape, dtype=bool)
        filtered_masks = []
        too_small_masks = []
        too_large_masks = []

        for mask in masks:
            bool_mask = mask.astype(bool)
            mask_size = np.sum(bool_mask)

            if min_pixels <= mask_size < max_pixels:
                combined_mask = combined_mask | bool_mask
                filtered_masks.append(mask)
            elif mask_size < min_pixels:
                too_small_masks.append(mask)
            else:
                too_large_masks.append(mask)

        # Invert the combined mask to get the background
        background_mask = ~combined_mask

        # Create background-only image
        background_image = original_image.copy()
        background_image[~background_mask] = [0, 0, 0]  # Set masked areas to black

        # Display background image
        ax2.imshow(background_image)
        ax2.set_title(f'Segments ({min_threshold:.1%} to {max_threshold:.1%} of image)')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            print(f"Visualization saved to {save_path}")

        plt.show()


def segment_and_get_part_mask(
        image_path,
        device="cpu",
        conf=0.25,
        iou=0.9,
        imgsz=1024,
        min_threshold=0.001,
        max_threshold=0.1
):
    """
    Performs image segmentation and calculates the part mask in a single method.

    Args:
        image_path (str): Path to the input image
        device (str): Device to run inference on ('cpu' or 'cuda')
        conf (float): Confidence threshold for segmentation
        iou (float): IoU threshold for segmentation
        imgsz (int): Input image size for segmentation
        min_threshold (float): Minimum size threshold as a fraction of image size (0.0 to 1.0)
        max_threshold (float): Maximum size threshold as a fraction of image size (0.0 to 1.0)

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The original image
            - object: The segmentation results
            - numpy.ndarray: The calculated part mask
            - numpy.ndarray: The calculated background mask
    """
    segmenter = ImageSegmenter()
    # Perform image segmentation
    original_image, results = segmenter.segment_image(
        image_path,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz
    )

    # Calculate part and background masks
    part_mask, background_mask = segmenter.calculate_background(
        original_image,
        results,
        min_threshold=min_threshold,
        max_threshold=max_threshold
    )

    part_mask = part_mask.astype(np.uint8) * 255
    background_mask = background_mask.astype(np.uint8) * 255

    return results, part_mask, background_mask





def main():
   
    # Example usage
    segmenter = ImageSegmenter()

    # Define image path
    image_path = "../data/dummy/part_2/part_3.png"

    # Segment image
    original_image, results = segmenter.segment_image(
        image_path,
        device="cpu",
        conf=0.25,
        iou=0.9
    )

    # Visualize all segments
    segmenter.visualize_all_segments(
        original_image,
        results
    )

    # Visualize background with size thresholds
    segmenter.visualize_background(
        original_image,
        results,
        min_threshold=0.001,  # Only use segments between 2% and 10% of image size
        max_threshold=0.05
    )

    # After calculating the background
    part_mask, background_mask = segmenter.calculate_background(original_image, results)

    # Visualize the results
    segmenter.visualize_segmented_part(original_image, part_mask, background_mask, save_path=None)


if __name__ == "__main__":
    main()