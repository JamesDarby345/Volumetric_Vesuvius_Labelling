import os
import blosc2
import numpy as np
import napari
from scipy import ndimage
import time

import numpy as np
from scipy import ndimage
import blosc2
import os

def combine_labeled_chunks(labeled_4d_array):
    if labeled_4d_array.size == 0:
        return None
    time_steps, depth, height, width = labeled_4d_array.shape
    combined_3d_array = np.zeros((depth, height, width), dtype=int)
    removed_labels = set()

    for t in range(time_steps):
        current_chunk = labeled_4d_array[t]
        current_labels = np.unique(current_chunk[current_chunk != 0])

        for label in current_labels:
            if label in removed_labels:
                continue

            label_mask = current_chunk == label
            overlap = np.any(combined_3d_array[label_mask] > 0)

            if overlap:
                overlap_size = np.sum(combined_3d_array[label_mask] > 0)
                label_size = np.sum(label_mask)

                if label_size > overlap_size:
                    overlapping_labels = np.unique(combined_3d_array[label_mask])
                    for olap_label in overlapping_labels[overlapping_labels != 0]:
                        combined_3d_array[combined_3d_array == olap_label] = 0
                        removed_labels.add(olap_label)
                    combined_3d_array[label_mask] = label
                else:
                    removed_labels.add(label)
            else:
                combined_3d_array[label_mask] = label

    return combined_3d_array

def read_origin_file(file_path):
    with open(file_path, 'r') as f:
        return np.array([float(coord.strip()) for coord in f])

def find_b2nd_files(root_directory):
    b2nd_files, origin_files = [], []
    for dirpath, _, filenames in os.walk(root_directory):
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
    print(f"Processing file {file_path}")
    # if not os.path.exists(file_path):
    #     print(f"Error: File {file_path} does not exist")
    #     return None
    nd_array = blosc2.open(file_path)
    rel_z, rel_y, rel_x = z - int(origin[2]), y - int(origin[1]), x - int(origin[0])

    if any(coord + padding >= shape or coord + chunk_size + padding <= -padding
           for coord, shape in zip((rel_z, rel_y, rel_x), nd_array.shape[::-1])):
        return None

    slice_coords = [
        (max(0, rel - padding), min(rel + chunk_size + padding, shape))
        for rel, shape in zip((rel_z, rel_y, rel_x), nd_array.shape[::-1])
    ]

    chunk = np.array(nd_array[slice_coords[2][0]:slice_coords[2][1],
                              slice_coords[1][0]:slice_coords[1][1],
                              slice_coords[0][0]:slice_coords[0][1]])
    chunk = np.swapaxes(chunk, 0, 2)

    pad_before = [max(0, padding - (rel - start)) for rel, start in zip((rel_z, rel_y, rel_x), [coord[0] for coord in slice_coords])]
    pad_after = [max(0, (chunk_size + 2 * padding) - chunk.shape[i] - pad_before[i]) for i in range(3)]

    padded_chunk = np.pad(chunk, list(zip(pad_before, pad_after)), mode='constant', constant_values=0)

    return padded_chunk if np.sum(padded_chunk) != 0 else None

def split_connected_components(chunk, chunk_i):
    labeled_array, num_features = ndimage.label(chunk)
    split_chunks = []
    for label in range(1, num_features + 1):
        component = (labeled_array == label)
        if np.sum(component) > 0:
            component_array = np.zeros_like(chunk)
            component_array[component] = chunk_i
            split_chunks.append(component_array)
            chunk_i += 1
    return split_chunks, chunk_i

def process_b2nd_files(root_directory, z, y, x, chunk_size, padding):
    b2nd_files, origin_files = find_b2nd_files(root_directory)
    chunks = []
    chunk_i = 1

    for b2nd_file, origin_file in zip(b2nd_files, origin_files):
        origin = read_origin_file(origin_file)
        chunk = read_b2nd_chunk(b2nd_file, origin, z, y, x, chunk_size, padding)
        if chunk is not None:
            split_chunks, chunk_i = split_connected_components(chunk, chunk_i)
            chunks.extend(split_chunks)

    result_array = np.stack(chunks, axis=0) if chunks else np.array([])
    return combine_labeled_chunks(result_array)

if __name__ == '__main__':
    current_directory = os.getcwd()
    root_directory = f"{current_directory}/data/fake/manual_sheet_segmentation/s1"
    zyx = '01744_01744_04048'
    z, y, x = map(int, zyx.split('_'))
    chunk_size = 256  # Size of the chunk
    padding = 20  # Padding parameter
    stime = time.time()
    result = process_b2nd_files(root_directory, z, y, x, chunk_size, padding)
    print(f"Elapsed time: {time.time() - stime}")

    viewer = napari.Viewer()
    if result is not None:
        viewer.add_labels(result, name="Combined Results")
    napari.run()