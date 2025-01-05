import sys
import os
import csv
import random
import matplotlib.pyplot as plt

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

            # Generate random values for x, y, and angle
            x = random.randint(50, 500)
            y = random.randint(50, 500)
            angle = random.choice([0, 45, 90, 135, 180])

            # Add the result to output data
            output_data.append({
                'part': part,
                'gripper': gripper,
                'x': x,
                'y': y,
                'angle': angle
                #'visualization': visualization_path
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



if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 solution/csvBuilder.py path/to/input/input.csv path/to/output/folder/")
        sys.exit(1)
        
input_csv_path = sys.argv[1] # (path/to/input/input.csv)

output_folder_path = sys.argv[2] # (path/to/output/folder/)

    # Run the solution
generate_results(input_csv_path, output_folder_path)