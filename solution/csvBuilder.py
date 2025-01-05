import sys
import os
import csv
import random
import matplotlib.pyplot as plt
import Preprocessing_Gripper as gripper
import processing_canny as part
import optimization as opt

def compute_solution(inputPath, outputPath):
    binary_image, binary_image_invert, shape = part.process_image(inputPath, outputPath, show=False, inverted_binary=False)
    _, _, centerTuple, _ = part.find_center(binary_image)
    combined_image = part.combine(binary_image, centerTuple, show=True)
    print("Shape of the image (x,y):", shape)

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
    compute_solution("../data/dummy/part_1/part_1.png", "soultion.png")