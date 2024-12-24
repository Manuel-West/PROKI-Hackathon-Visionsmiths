import sys
import os
import csv
import random
import matplotlib.pyplot as plt

def generate_results(input_csv, output_folder):
    """
    Main function to generate results based on an input semicolon-delimited CSV file.

    :param input_csv: Path to the input CSV file containing 'part' and 'gripper' columns.
    :param output_folder: Path to the folder where results (CSV and visualizations) will be saved.
    """
    try:
        # Read the input CSV file with semicolon delimiter
        with open(input_csv, 'r') as file:
            reader = csv.DictReader(file, delimiter=';')  # Specify semicolon delimiter
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

            # Add visualization (optional)
            #visualization_path = os.path.join(output_folder, f"vis_{part.split('.')[0]}_{gripper.split('.')[0]}.png")
            #create_visualization(part, gripper, x, y, angle, visualization_path)

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
            writer = csv.DictWriter(file, fieldnames=['part', 'gripper', 'x', 'y', 'angle', 'visualization'])
            writer.writeheader()
            writer.writerows(output_data)

        print("Processing completed successfully!")
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)



'''
def create_visualization(part, gripper, x, y, angle, output_path):
    """
    Generate a simple visualization of the part and gripper placement.

    :param part: Name of the part file (for display purposes).
    :param gripper: Name of the gripper file (for display purposes).
    :param x: X-coordinate of the placement.
    :param y: Y-coordinate of the placement.
    :param angle: Rotation angle of the gripper.
    :param output_path: Path to save the generated visualization image.
    """
    # Create a figure and axis using Matplotlib
    fig, ax = plt.subplots()
    ax.set_title(f"Part: {part}, Gripper: {gripper}")
    ax.scatter(x, y, label=f"Position: ({x}, {y}), Angle: {angle}°", color='red')
    ax.legend()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    plt.grid(True)

    # Save the visualization image to the specified path
    plt.savefig(output_path)
    plt.close()
'''




if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python solution/main.py path/to/input/tasks.csv path/to/output/folder")
        sys.exit(1)
        
'''
    /Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/input.csv
    
    /Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/csvBuilder.py
    
    /Users/jonasludwig/Desktop/Hackathon/PROKI-Hackathon-Visionsmiths/solution/

'''     

input_csv_path = sys.argv[1] #path to input

output_folder_path = sys.argv[2] #path to output folder 

    # Run the solution
generate_results(input_csv_path, output_folder_path)