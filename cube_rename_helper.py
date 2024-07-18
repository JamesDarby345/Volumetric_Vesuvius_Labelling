import os
import re

def rename_folders_and_files(root_path):
    # Walk through all directories and files in the given path
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Check for folders named cube_z_y_x
        for i, dirname in enumerate(dirnames):
            match = re.match(r'cube_(\d+)_(\d+)_(\d+)', dirname)
            if match:
                z, y, x = match.groups()
                new_dirname = f"{z.zfill(5)}_{y.zfill(5)}_{x.zfill(5)}"
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)
                os.rename(old_path, new_path)
                print(f"Renamed folder: {old_path} -> {new_path}")
                # Update dirname in the list so we use the new name for file operations
                dirnames[i] = new_dirname

        # Check all files in the current directory for renaming
        for filename in filenames:
            if filename.endswith('.nrrd'):
                match = re.match(r'(volume|mask)_(\d+)_(\d+)_(\d+)\.nrrd', filename)
                if match:
                    file_type, z, y, x = match.groups()
                    new_filename = f"{z.zfill(5)}_{y.zfill(5)}_{x.zfill(5)}_{file_type}.nrrd"
                    old_file_path = os.path.join(dirpath, filename)
                    new_file_path = os.path.join(dirpath, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed file: {old_file_path} -> {new_file_path}")


# Usage
root_folder_path = "/Users/jamesdarby/Desktop/manually_labelled_cubes"
rename_folders_and_files(root_folder_path)