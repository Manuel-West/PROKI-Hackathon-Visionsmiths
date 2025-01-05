import sys
import os
import csv
import random
import matplotlib.pyplot as plt
import Preprocessing_Gripper as gripper
import processing_canny as part
import optimization as opt
import torch as torch
from torch.autograd import Variable as V

def compute_solution(part_input_path, gripper_input_path, output_path, show=False):
    part_mask, binary_image_invert, shape = part.process_image(part_input_path, output_path, show=False, inverted_binary=True)
    _, _, centerTuple, _ = part.find_center(part_mask)
    combined_image = part.combine(part_mask, centerTuple, show)

    gripper_mask, cX, cY = gripper.preprocessing_gripper(gripper_input_path, shape[0], shape[1], show)
    assert part_mask.shape == gripper_mask.shape
    gripper_mask_torch = V(torch.tensor(gripper_mask).double(), True)
    #part_mask_torch = V(torch.tensor(part_mask).double(), True)

    random_values = torch.rand_like(torch.tensor(part_mask).float()) * 0.8 + 0.2  # Scale to [0.2, 1.0]

    # Only apply random values where part_mask is non-zero
    new_part_mask = torch.where(torch.tensor(part_mask) > 0, random_values, torch.zeros_like(random_values))

    # Convert to double and make it require gradients
    part_mask_torch = torch.tensor(new_part_mask, dtype=torch.float64, requires_grad=True)

    x, x_hist = opt.optimize_coordinates(template=gripper_mask_torch, reference=part_mask_torch, target_coords=centerTuple, max_iter=100, tol=1e-1, show=True)
    print(x)
    #return solution

def generate_results(input_csv, output_folder, delimeter= ';'):
    """
    Main function to generate results based on an input semicolon-delimited CSV file.

    :param input_csv: Path to the input CSV file containing 'part' and 'gripper' columns.
    :param output_folder: Path to the folder where results (CSV and visualizations) will be saved.
    :param: delimeter: default delimeter is ' ; ' -> use delimeter= ',' if other  
    """
    try:
        # Read the input CSV file with (default) semicolon delimiter
        with open(input_csv, 'r') as file:
            reader = csv.DictReader(file, delimiter= delimeter)  # Specify semicolon delimiter
            rows = list(reader)

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Prepare output data
        output_data = []
        for row in rows:
            part = row['part']
            gripper = row['gripper']



            # Add the result to output data
            output_data.append({
                'part': part,
                'gripper': gripper,
                'x': x,
                'y': y,
                'angle': angle
            })

        # Write output CSV file with comma delimiter
        output_csv = os.path.join(output_folder, "solutions.csv")
        with open(output_csv, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['part', 'gripper', 'x', 'y', 'angle'])
            writer.writeheader()
            writer.writerows(output_data)

        print("Processing completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 solution/csvBuilder.py path/to/input/input.csv path/to/output/folder/")
        sys.exit(1)


    input_csv_path = sys.argv[1]  # (path/to/input/input.csv)

    output_folder_path = sys.argv[2]  # (path/to/output/folder/)

    # Run the solution
    generate_results(input_csv_path, output_folder_path)

if __name__ == "__main__":
    compute_solution("../data/dummy/part_1/part_1.png", "../data/dummy/part_1/gripper_2.png","soultion.png",False)