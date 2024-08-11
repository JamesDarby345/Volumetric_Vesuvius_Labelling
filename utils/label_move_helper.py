import os
import shutil
import re

def process_files(source_folder):
    # Regular expression pattern to match the desired file names
    pattern = r'^(\d{5})_(\d{5})_(\d{5}).*_label\.nrrd$'

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        match = re.match(pattern, filename)
        if match:
            # Extract z, y, x values from the filename
            z, y, x = match.groups()

            # Create the new folder name
            new_folder_name = f"{z}_{y}_{x}"

            # Create the full path for the new folder
            new_folder_path = os.path.join(os.path.dirname(source_folder), new_folder_name)

            # Create the new folder if it doesn't exist
            os.makedirs(new_folder_path, exist_ok=True)

            # Move the file to the new folder
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(new_folder_path, filename)
            shutil.move(source_file, destination_file)

            print(f"Moved {filename} to {new_folder_path}")

# Usage
source_folder = "../output/volumetric_labels_s1/!dump"
process_files(source_folder)