import os
import blosc2
import numpy as np
import napari
from scipy import ndimage

def combine_labeled_chunks(labeled_4d_array):
    # Get the shape of the 4D array
    time_steps, depth, height, width = labeled_4d_array.shape
    
    # Initialize the output 3D array
    combined_3d_array = np.zeros((depth, height, width), dtype=int)
    
    # Keep track of removed labels
    removed_labels = set()
    
    # Iterate through each time step
    for t in range(time_steps):
        # Get the current 3D chunk
        current_chunk = labeled_4d_array[t]
        
        # Find unique labels in the current chunk
        current_labels = np.unique(current_chunk)
        current_labels = current_labels[current_labels != 0]  # Exclude background
        
        for label in current_labels:
            if label in removed_labels:
                continue
            
            # Create a mask for the current label
            label_mask = current_chunk == label
            
            # Check for overlap
            overlap = np.any(combined_3d_array[label_mask] > 0)
            
            if overlap:
                # If there's overlap, compare the sizes
                overlap_size = np.sum(combined_3d_array[label_mask] > 0)
                label_size = np.sum(label_mask)
                
                if label_size > overlap_size:
                    # Current label is larger, remove overlapping labels from combined array
                    overlapping_labels = np.unique(combined_3d_array[label_mask])
                    for olap_label in overlapping_labels:
                        if olap_label != 0:
                            combined_3d_array[combined_3d_array == olap_label] = 0
                            removed_labels.add(olap_label)
                    # Add current label
                    combined_3d_array[label_mask] = label
                else:
                    # Current label is smaller or equal, don't add it
                    removed_labels.add(label)
            else:
                # No overlap, simply add the label
                combined_3d_array[label_mask] = label
    
    return combined_3d_array

def read_origin_file(file_path):
    with open(file_path, 'r') as f:
        origin = f.readlines()
    return np.array([float(coord.strip()) for coord in origin])

def read_b2nd_files(root_directory):
    b2nd_files = []
    origin_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.b2nd'):
                b2nd_file = os.path.join(dirpath, filename)
                origin_file = os.path.join(dirpath, f"{filename.split('.')[0]}_origin.txt")
                
                if os.path.exists(origin_file):
                    b2nd_files.append(b2nd_file)
                    origin_files.append(origin_file)
                else:
                    print(f"Warning: No matching origin file found for {b2nd_file}")
    
    return b2nd_files, origin_files

def read_b2nd_chunk(file_path, origin, z, y, x, chunk_size, padding):
    nd_array = blosc2.open(file_path)
    print(f"Processing: {file_path}")
    print(f"B2nd file shape xyz: {nd_array.shape}")
    
    print(f"Origin xyz: {origin}")
    print(f"Requested start coordinates (absolute) zyx: {z}, {y}, {x}")
    
    # Calculate coordinates relative to the b2nd file
    rel_z, rel_y, rel_x = z - int(origin[2]), y - int(origin[1]), x - int(origin[0])
    
    print(f"Start coordinates (relative to b2nd) zyx: {rel_z}, {rel_y}, {rel_x}")

    # Check if the chunk (including padding) is completely out of bounds
    if rel_z + padding >= nd_array.shape[2] or rel_y + padding >= nd_array.shape[1] or rel_x + padding >= nd_array.shape[0] or \
       rel_z + chunk_size + padding <= -padding or rel_y + chunk_size + padding <= -padding or rel_x + chunk_size + padding <= -padding:
        print("Chunk is completely out of bounds, skipping...")
        return None

    # Calculate the slice indices in b2nd file coordinates, including padding
    z_start, y_start, x_start = max(0, rel_z - padding), max(0, rel_y - padding), max(0, rel_x - padding)
    z_end = min(rel_z + chunk_size + padding, nd_array.shape[2])
    y_end = min(rel_y + chunk_size + padding, nd_array.shape[1])
    x_end = min(rel_x + chunk_size + padding, nd_array.shape[0])

    print(f"Slice coordinates (in b2nd file) zyx start: {z_start}, {y_start}, {x_start}")
    print(f"Slice coordinates (in b2nd file) zyx end: {z_end}, {y_end}, {x_end}")
    
    # Extract the chunk and convert to numpy array
    chunk = np.array(nd_array[x_start:x_end, y_start:y_end, z_start:z_end])
    
    # Swap axes from x,y,z to z,y,x
    chunk = np.swapaxes(chunk, 0, 2)
    
    print(f"Extracted chunk shape: {chunk.shape}")
    
    # Calculate padding
    pad_before = [max(0, padding - (rel_z - z_start)),
                  max(0, padding - (rel_y - y_start)),
                  max(0, padding - (rel_x - x_start))]
    pad_after = [
        max(0, (chunk_size + 2 * padding) - chunk.shape[0] - pad_before[0]),
        max(0, (chunk_size + 2 * padding) - chunk.shape[1] - pad_before[1]),
        max(0, (chunk_size + 2 * padding) - chunk.shape[2] - pad_before[2])
    ]
    
    print(f"Padding before zyx: {pad_before}")
    print(f"Padding after zyx: {pad_after}")
    
    # Pad the chunk to ensure consistent size
    padded_chunk = np.pad(chunk, list(zip(pad_before, pad_after)), mode='constant', constant_values=0)
    
    print(f"Padded chunk shape: {padded_chunk.shape}")
    print(f"Sum: {np.sum(padded_chunk)}, Unique values: {np.unique(padded_chunk)}")
    
    if np.sum(padded_chunk) == 0:
        print("Chunk sum is 0, skipping...")
        return None
    
    return padded_chunk

def split_connected_components(chunk, chunk_i):
    # Perform connected component analysis
    labeled_array, num_features = ndimage.label(chunk)
    
    # Split the chunk into separate arrays for each connected component
    split_chunks = []
    for label in range(1, num_features + 1):
        component = (labeled_array == label)
        if np.sum(component) > 0:  # Only process non-empty components
            # Create a new 3D array for this component
            component_array = np.zeros_like(chunk)
            component_array[component] = chunk_i
            split_chunks.append(component_array)
            chunk_i += 1  # Increment chunk_i for the next label
    
    return split_chunks, chunk_i

def process_b2nd_files(root_directory, z, y, x, chunk_size, padding):
    b2nd_files, origin_files = read_b2nd_files(root_directory)
    chunks = []
    
    chunk_i = 1
    for b2nd_file, origin_file in zip(b2nd_files, origin_files):
        origin = read_origin_file(origin_file)
        chunk = read_b2nd_chunk(b2nd_file, origin, z, y, x, chunk_size, padding)
        if chunk is not None:
            # Split the chunk into connected components
            split_chunks, chunk_i = split_connected_components(chunk, chunk_i)
            chunks.extend(split_chunks)
    
    # Only stack if we have valid chunks
    if chunks:
        result_array = np.stack(chunks, axis=0)
    else:
        result_array = np.array([])
    
    result_array = combine_labeled_chunks(result_array)
    return result_array

# Example usage:
current_directory = os.getcwd()
root_directory = f"{current_directory}/data/manual_sheet_segmentation/s1"
zyx = '01744_01744_04048'
z, y, x = map(int, zyx.split('_'))
chunk_size = 256  # Size of the chunk
padding = 20  #Padding parameter

result = process_b2nd_files(root_directory, z, y, x, chunk_size, padding)

viewer = napari.Viewer()
viewer.add_labels(result, name="Combined Results")
napari.run()