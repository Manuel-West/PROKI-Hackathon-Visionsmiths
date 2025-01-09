import sys
import os
import csv
import random

import cv2
import matplotlib.pyplot as plt
import Preprocessing_Gripper as gripper
import processing_canny as part
import optimization as opt
import torch as torch
from torch.autograd import Variable as V
from pathlib import Path
from argparse import ArgumentParser
import math


def compute_solution(part_input_path, gripper_input_path, output_path, show=False) -> tuple:
    """
    Processes the part and gripper images, aligns the gripper to the part, and computes the solution.

    :param part_input_path: Path to the input image of the part.
    :param gripper_input_path: Path to the input image of the gripper.
    :param output_path: Path where intermediate results will be saved.
    :param show: Boolean flag to display visualizations of intermediate steps.
    :return: Tuple (solution_x, solution_y, solution_alpha).
    """
    # Process the part image to get its mask, shape, and center
    part_mask, binary_image_invert, shape = part.process_image(part_input_path, output_path, show=False, inverted_binary=True, save=True)

    # add padding to the borders
    padding = 50
    part_mask = cv2.copyMakeBorder(part_mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 255)

    # calculate center
    _, _, centerTuple, _ = part.find_center(part_mask)

    # Preprocess the gripper image and ensure it matches the shape of the part
    gripper_mask, cX, cY = gripper.preprocessing_gripper(gripper_input_path, part_mask.shape[1], part_mask.shape[0], show)

    # Ensure the dimensions of part_mask and gripper_mask are the same
    assert part_mask.shape == gripper_mask.shape, f"Mask shapes must match: part_mask {part_mask.shape} != gripper_mask {gripper_mask.shape}"

    # Convert the gripper mask to a PyTorch tensor
    gripper_mask_torch = V(torch.tensor(gripper_mask).double(), True)

    # Create a random mask for the part, scaled between 0.2 and 1.0
    random_values = torch.rand_like(torch.tensor(part_mask).float()) * 0.8 + 0.2
    new_part_mask = torch.where(torch.tensor(part_mask) > 0, random_values, torch.zeros_like(random_values))

    # Convert the new part mask to a PyTorch tensor with gradient tracking enabled
    part_mask_torch = torch.tensor(new_part_mask, dtype=torch.float64, requires_grad=True)

    # Use optimization to align the gripper mask with the part mask
    x, x_hist = opt.optimize_coordinates(template=gripper_mask_torch, reference=part_mask_torch, target_coords=centerTuple, max_iter=100, tol=1e-8, show=True, output_path=output_path)
    print(x)
    # Compute the solution coordinates and angle based on the optimization results
    solution_x = x[0]
    solution_y = x[1]
    solution_alpha = x[2]
    solution = (solution_x, solution_y, solution_alpha)

    return solution


def generate_results(input_csv, output_filename, delimiter=';'):
    """
    Reads an input CSV file containing part and gripper image paths, computes solutions, and writes results to an output CSV.

    :param input_csv: Path to the input CSV file.
    :param output_filename: Path to the output CSV file to save results.
    :param delimiter: Delimiter used in the input CSV file (default is ';').
    """
    try:
        # Read the input CSV file using the specified delimiter
        with open(input_csv, 'r') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            rows = list(reader)

        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Prepare the output data for writing
        output_data = []
        for row in rows:
            part_path = row['part']  # Path to the part image
            gripper_path = row['gripper']  # Path to the gripper image

            # Compute the solution for the given part and gripper
            part_name = os.path.splitext(os.path.basename(part_path))[0]
            gripper_name = os.path.splitext(os.path.basename(gripper_path))[0]

            # Construct output folder and ensure a single .png extension
            output_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                f"solution_{part_name}_{gripper_name}.png"
            )
            
            print("Output folder:", output_folder)
            solution = compute_solution(part_path, gripper_path, output_folder, show=False)

            # Append the result to the output data
            output_data.append({
                'part': part_path,
                'gripper': gripper_path,
                'x': math.ceil(solution[0]) if solution[0] % 1 >= 0.5 else math.floor(solution[0]),
                'y': math.ceil(solution[1]) if solution[1] % 1 >= 0.5 else math.floor(solution[1]),
                'angle': (solution[2] % 360 + 360) % 360
            })

        # Write the results to the output CSV file
        with open(output_filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['part', 'gripper', 'x', 'y', 'angle'])
            writer.writeheader()
            writer.writerows(output_data)

        print("Processing completed successfully!")
        sys.exit(0)  # Exit the program successfully

    except Exception as e:
        # Print an error message and exit with a failure code
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """
    Main entry point of the script. Parses command-line arguments and generates results.
    """
    # Parse command-line arguments
    parser = ArgumentParser(prog="csvBuilder",
                            description="Creates a CSV file containing results (part, gripper, x, y, angle) for an input CSV with part and gripper paths.")
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("output", help="Path to the output CSV file.")
    args = parser.parse_args()

    # Ensure the input CSV file exists
    input_csv_path = args.input
    output_folder_path = args.output
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    # Generate results based on the input and output paths
    generate_results(input_csv_path, output_folder_path, delimiter=',')


if __name__ == "__main__":
    main() 