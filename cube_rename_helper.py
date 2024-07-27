import os
import re

def rename_folders_and_files(root_path):
    # Walk through all directories and files in the given path
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Remove .DS_Store files
        if '.DS_Store' in filenames:
            ds_store_path = os.path.join(dirpath, '.DS_Store')
            os.remove(ds_store_path)
            print(f"Removed file: {ds_store_path}")
        
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
                # Match the original pattern
                match = re.match(r'(volume|mask)_(\d+)_(\d+)_(\d+)\.nrrd', filename)
                if match:
                    file_type, z, y, x = match.groups()
                    new_filename = f"{z.zfill(5)}_{y.zfill(5)}_{x.zfill(5)}_{file_type}.nrrd"
                else:
                    # Match the new pattern
                    match = re.match(r'(\d+)_(\d+)_(\d+)_zyx_\d+_chunk_[a-zA-Z0-9_]+_vol_(raw|label)\.nrrd', filename)
                    if match:
                        z, y, x, file_type = match.groups()
                        file_type = 'volume' if file_type == 'raw' else 'mask'
                        new_filename = f"{z.zfill(5)}_{y.zfill(5)}_{x.zfill(5)}_{file_type}.nrrd"
                    else:
                        # Match z_y_x_mask or z_y_x_volume pattern without leading zeros
                        match = re.match(r'(\d+)_(\d+)_(\d+)_(mask|volume)\.nrrd', filename)
                        if match:
                            z, y, x, file_type = match.groups()
                            new_filename = f"{z.zfill(5)}_{y.zfill(5)}_{x.zfill(5)}_{file_type}.nrrd"
                        else:
                            continue  # Skip files that don't match any pattern

                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} -> {new_file_path}")

# Usage
root_folder_path = "/Users/jamesdarby/Desktop/manually_labelled_cubes"
rename_folders_and_files(root_folder_path)