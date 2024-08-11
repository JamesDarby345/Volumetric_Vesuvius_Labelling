import os
import shutil

def sync_directories(source_path, destination_path):
    # Ensure both paths exist
    if not os.path.exists(source_path) or not os.path.exists(destination_path):
        raise ValueError("Both source and destination paths must exist.")

    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        # Calculate the relative path
        relative_path = os.path.relpath(root, source_path)
        destination_dir = os.path.join(destination_path, relative_path)

        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print(f"Created directory: {destination_dir}")

        # Copy files
        for file in files:
            source_file = os.path.join(root, file)
            dest_file = os.path.join(destination_dir, file)
            shutil.copy2(source_file, dest_file)

if __name__ == "__main__":
    source_path = '/Users/jamesdarby/Desktop/semantic_masks'
    destination_path = '/Users/jamesdarby/Documents/VesuviusScroll/GP/Volumetric_Vesuvius_Labelling/data/nrrd_cubes/s1'

    try:
        sync_directories(source_path, destination_path)
        print("Synchronization completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")