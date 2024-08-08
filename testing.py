import os
import blosc2
import numpy as np
import napari

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

def read_b2nd_chunk(file_path, origin, z, y, x, chunk_size):
    nd_array = blosc2.open(file_path)
    print(f"Processing: {file_path}")
    print(f"B2nd file shape xyz: {nd_array.shape}")
    
    print(f"Origin xyz: {origin}")
    print(f"Requested start coordinates (absolute) zyx: {z}, {y}, {x}")
    
    # Calculate coordinates relative to the b2nd file
    rel_z, rel_y, rel_x = z - int(origin[2]), y - int(origin[1]), x - int(origin[0])
    
    print(f"Start coordinates (relative to b2nd) zyx: {rel_z}, {rel_y}, {rel_x}")

    # Check if the chunk is completely out of bounds
    if rel_z >= nd_array.shape[2] or rel_y >= nd_array.shape[1] or rel_x >= nd_array.shape[0] or \
       rel_z + chunk_size <= 0 or rel_y + chunk_size <= 0 or rel_x + chunk_size <= 0:
        print("Chunk is completely out of bounds, skipping...")
        return None

    # Calculate the slice indices in b2nd file coordinates
    z_start, y_start, x_start = max(0, rel_z), max(0, rel_y), max(0, rel_x)
    z_end = min(rel_z + chunk_size, nd_array.shape[2])
    y_end = min(rel_y + chunk_size, nd_array.shape[1])
    x_end = min(rel_x + chunk_size, nd_array.shape[0])

    print(f"Slice coordinates (in b2nd file) zyx start: {z_start}, {y_start}, {x_start}")
    print(f"Slice coordinates (in b2nd file) zyx end: {z_end}, {y_end}, {x_end}")
    
    # Extract the chunk and convert to numpy array
    chunk = np.array(nd_array[x_start:x_end, y_start:y_end, z_start:z_end])
    
    # Swap axes from x,y,z to z,y,x
    chunk = np.swapaxes(chunk, 0, 2)
    
    print(f"Extracted chunk shape: {chunk.shape}")
    
    # Calculate padding
    pad_before = [max(0, -rel_z), max(0, -rel_y), max(0, -rel_x)]
    pad_after = [
        max(0, (rel_z + chunk_size) - nd_array.shape[2]),
        max(0, (rel_y + chunk_size) - nd_array.shape[1]),
        max(0, (rel_x + chunk_size) - nd_array.shape[0])
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

def process_b2nd_files(root_directory, z, y, x, chunk_size):
    b2nd_files, origin_files = read_b2nd_files(root_directory)
    chunks = []
    
    for b2nd_file, origin_file in zip(b2nd_files, origin_files):
        origin = read_origin_file(origin_file)
        chunk = read_b2nd_chunk(b2nd_file, origin, z, y, x, chunk_size)
        if chunk is not None:
            chunks.append(chunk)
    
    # Only stack if we have valid chunks
    if chunks:
        result_array = np.stack(chunks, axis=0)
    else:
        result_array = np.array([])
    
    return result_array

# Example usage:
current_directory = os.getcwd()
root_directory = f"{current_directory}/data/manual_sheet_segmentation/s1"
zyx = '01744_01744_04048'
z, y, x = map(int, zyx.split('_'))
chunk_size = 256  # Size of the chunk

result = process_b2nd_files(root_directory, z, y, x, chunk_size)
print(f"Final result shape: {result.shape}")

for i in range(result.shape[0]):
    print(f"Chunk {i}:")
    print(f"Sum: {np.sum(result[i])}, Unique values: {np.unique(result[i])}")
    print()

viewer = napari.Viewer()
for i in range(result.shape[0]):    
    viewer.add_labels(result[i], name=f"Chunk {i}")
napari.run()