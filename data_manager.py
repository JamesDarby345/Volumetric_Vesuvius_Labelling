# data_manager.py

import os
import nrrd
import numpy as np
import zarr
from dask import array as da
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
import asyncio
from scipy import ndimage
import blosc2
from helper import *

class DataManager:
    def __init__(self, cube_config):
        self.cube_config = cube_config
        self.raw_data = None
        self.padded_raw_data = None
        self.original_label_data = None
        self.label_data = None
        self.original_ink_pred_data = None
        self.voxelized_segmentation_mesh_data = None
        self.label_header = None
        self.raw_data_header = None
        self.raw_data_zarr_shape = None
        self.is_saving = False

        self.load_data()

    def reload_data(self):
        print(f"Reloading data in data manager {self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}")
        # Reset all data attributes
        self.raw_data = None
        self.padded_raw_data = None
        self.original_label_data = None
        self.label_data = None
        self.original_ink_pred_data = None
        self.voxelized_segmentation_mesh_data = None
        self.label_header = None

        # Reload all data
        self.load_data()

    def load_data(self):
        self.load_raw_data()
        self.load_label_data()
        self.load_ink_pred_data()
        self.load_voxelized_segmentation_mesh_data()
            
    def load_voxelized_segmentation_mesh_data(self):
        saved_mesh_file_path = self.get_seg_mesh_file_path(self.cube_config.z, self.cube_config.y, self.cube_config.x)

        if os.path.exists(saved_mesh_file_path):
            self.voxelized_segmentation_mesh_data, _ = nrrd.read(saved_mesh_file_path)
            print(f"Loaded segmentation mesh from {saved_mesh_file_path}")
        else:
            self.voxelized_segmentation_mesh_data = self.process_b2nd_files(self.cube_config.voxelised_mesh_path, self.cube_config.z_num, self.cube_config.y_num, self.cube_config.x_num, self.cube_config.chunk_size, self.cube_config.voxelized_mesh_pad_amount)

    def load_raw_data(self):
        if not self.cube_config.using_raw_data_zarr:
            volume_file_path = os.path.join(self.cube_config.nrrd_cube_path, 
                                            f'{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}', 
                                            f'{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}_volume.nrrd')
            self.raw_data, _ = nrrd.read(volume_file_path)
            self.padded_raw_data = self.get_padded_nrrd_data()
        else:
            raw_data_zarr_multi_res = zarr.open(self.cube_config.raw_data_zarr_path, mode='r')
            self.raw_data_zarr_shape = raw_data_zarr_multi_res[0].shape
            self.raw_data = raw_data_zarr_multi_res[0][self.cube_config.z_num:self.cube_config.z_num+self.cube_config.chunk_size, 
                                                       self.cube_config.y_num:self.cube_config.y_num+self.cube_config.chunk_size, 
                                                       self.cube_config.x_num:self.cube_config.x_num+self.cube_config.chunk_size]
            self.padded_raw_data = self.get_padded_data_zarr(raw_data_zarr_multi_res[0])

    def create_papyrus_mask(self, nrrd_cube_folder_path, mask_file_path):
        self.original_label_data = self.threshold_mask(self.raw_data, factor=self.cube_config.factor).astype(np.uint8)
        os.makedirs(nrrd_cube_folder_path, exist_ok=True)
        print("Creating Papyrus Label from thresholded raw data, may take a few seconds...")
        nrrd.write(mask_file_path, self.original_label_data)
        self.original_label_data, self.label_header = nrrd.read(mask_file_path)

    def load_label_data(self):
        output_folder_path = os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.cube_config.scroll_name}')
        saved_label_file_path = os.path.join(output_folder_path, 
                                             f"{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}",
                                             f"{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}_zyx_{self.cube_config.chunk_size}_chunk_{self.cube_config.scroll_name}_vol_label.nrrd")

        nrrd_cube_folder_path = os.path.join(self.cube_config.nrrd_cube_path, f'{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}')
        mask_file_path = os.path.join(nrrd_cube_folder_path, f'{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}_mask.nrrd')

        # Load or create original label data
        if os.path.exists(mask_file_path):
            print(f"Loading original label data from mask file {mask_file_path}")
            self.original_label_data, self.label_header = nrrd.read(mask_file_path)
        elif self.cube_config.create_papyrus_mask_if_not_provided:
            print("Creating papyrus mask as original label data")
            self.create_papyrus_mask(nrrd_cube_folder_path, mask_file_path)
        else:
            print("No mask file found and create_papyrus_mask_if_not_provided is False.")
            self.original_label_data = None

        # Load edited label data if it exists, otherwise use original label data
        if os.path.exists(saved_label_file_path):
            print(f"Loading edited label data from saved label {saved_label_file_path}")
            self.label_data, self.label_header = nrrd.read(saved_label_file_path)
        elif self.original_label_data is not None:
            print("No saved label file found. Using original label data for editing.")
            self.label_data = self.original_label_data.copy()
        else:
            print("No label data available for editing.")
            self.label_data = None

        if self.cube_config.smoother_labels:
            if self.original_label_data is not None:
                self.original_label_data = pad_array(self.original_label_data, self.cube_config.chunk_size)
            if self.label_data is not None:
                self.label_data = pad_array(self.label_data, self.cube_config.chunk_size)



    def load_ink_pred_data(self):
        saved_ink_pred_file_path = os.path.join(os.getcwd(), 'output', 
                                                f'volumetric_labels_{self.cube_config.scroll_name}', 
                                                f"{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}",
                                                f"{self.cube_config.z}_{self.cube_config.y}_{self.cube_config.x}_zyx_{self.cube_config.chunk_size}_chunk_{self.cube_config.scroll_name}_ink_label.nrrd")

        if os.path.exists(saved_ink_pred_file_path):
            self.original_ink_pred_data, _ = nrrd.read(saved_ink_pred_file_path)
        elif self.cube_config.using_ink_pred_zarr:
            ink_pred_zarr = zarr.open(self.cube_config.ink_pred_zarr_path, mode='r')
            ink_pred_dask = da.from_zarr(ink_pred_zarr)

            if self.cube_config.raw_data_axis_order is None or self.cube_config.ink_pred_label_axis_order is None and self.raw_data_zarr_shape is not None:
                transpose_params = self.get_transpose_params_from_shapes(self.raw_data_zarr_shape, ink_pred_dask.shape) 
                ink_pred_dask = ink_pred_dask.transpose(transpose_params)
            elif self.cube_config.raw_data_axis_order is not None and self.cube_config.ink_pred_label_axis_order is not None:
                transpose_params = self.get_transpose_params_from_axis_order(self.cube_config.raw_data_axis_order, self.cube_config.ink_pred_label_axis_order)
                ink_pred_dask = ink_pred_dask.transpose(transpose_params)
            else:
                print("Could not ensure alignment of ink prediction label with raw data, using default axis order. Please check the axis order of the raw data and ink prediction label and set it in the config file if incorrect")
            
            ink_pred_region = ink_pred_dask[self.cube_config.z_num:self.cube_config.z_num+self.cube_config.chunk_size, 
                                            self.cube_config.y_num:self.cube_config.y_num+self.cube_config.chunk_size, 
                                            self.cube_config.x_num:self.cube_config.x_num+self.cube_config.chunk_size]
            thresholded_data = np.array(da.where(ink_pred_region < self.cube_config.ink_threshold, 0, 1)).astype(np.uint8)
            self.original_ink_pred_data = thresholded_data

        if self.cube_config.smoother_labels and self.original_ink_pred_data is not None:
            self.original_ink_pred_data = pad_array(self.original_ink_pred_data, self.cube_config.chunk_size)

    def get_padded_nrrd_data(self):
        padded_raw_data = None
        missing_cubes = []
        
        neighbors = [
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
            (-1, 0, -1),  (-1, 0, 0),  (-1, 0, 1),
            (-1, 1, -1),  (-1, 1, 0),  (-1, 1, 1),
            (0, -1, -1),  (0, -1, 0),  (0, -1, 1),
            (0, 0, -1),   (0, 0, 0),   (0, 0, 1),
            (0, 1, -1),   (0, 1, 0),   (0, 1, 1),
            (1, -1, -1),  (1, -1, 0),  (1, -1, 1),
            (1, 0, -1),   (1, 0, 0),   (1, 0, 1),
            (1, 1, -1),   (1, 1, 0),   (1, 1, 1)
        ]
        
        padded_size = self.cube_config.chunk_size + 2 * self.cube_config.pad_amount
        padded_raw_data = np.zeros((padded_size, padded_size, padded_size))
        
        for dz, dy, dx in neighbors:
            neighbor_z = str(int(self.cube_config.z) + dz * self.cube_config.chunk_size).zfill(5)
            neighbor_y = str(int(self.cube_config.y) + dy * self.cube_config.chunk_size).zfill(5)
            neighbor_x = str(int(self.cube_config.x) + dx * self.cube_config.chunk_size).zfill(5)
            
            filename = f"{neighbor_z}_{neighbor_y}_{neighbor_x}/{neighbor_z}_{neighbor_y}_{neighbor_x}_volume.nrrd"
            filepath = os.path.join(self.cube_config.nrrd_cube_path, filename)
            
            if os.path.exists(filepath):
                try:
                    data, header = nrrd.read(filepath)
                except Exception as e:
                    print(f"An error occurred while reading {filepath}: {e}")
                    continue
                
                z_start = self.cube_config.chunk_size - self.cube_config.pad_amount if dz < 0 else 0
                z_end = self.cube_config.pad_amount if dz > 0 else self.cube_config.chunk_size
                y_start = self.cube_config.chunk_size - self.cube_config.pad_amount if dy < 0 else 0
                y_end = self.cube_config.pad_amount if dy > 0 else self.cube_config.chunk_size
                x_start = self.cube_config.chunk_size - self.cube_config.pad_amount if dx < 0 else 0
                x_end = self.cube_config.pad_amount if dx > 0 else self.cube_config.chunk_size

                extracted_data = data[z_start:z_end, y_start:y_end, x_start:x_end]
                
                z_pad_start = self.cube_config.pad_amount + (dz) * self.cube_config.pad_amount
                if dz == 1:
                    z_pad_start = self.cube_config.pad_amount + self.cube_config.chunk_size
                y_pad_start = self.cube_config.pad_amount + (dy) * self.cube_config.pad_amount
                if dy == 1:
                    y_pad_start = self.cube_config.pad_amount + self.cube_config.chunk_size
                x_pad_start = self.cube_config.pad_amount + (dx) * self.cube_config.pad_amount
                if dx == 1:
                    x_pad_start = self.cube_config.pad_amount + self.cube_config.chunk_size
                
                z_pad_end = z_pad_start + extracted_data.shape[0]
                y_pad_end = y_pad_start + extracted_data.shape[1]
                x_pad_end = x_pad_start + extracted_data.shape[2]

                padded_raw_data[z_pad_start:z_pad_end, y_pad_start:y_pad_end, x_pad_start:x_pad_end] = extracted_data
            else:
                missing_cubes.append((neighbor_z, neighbor_y, neighbor_x))
        
        if missing_cubes:
            print("List of missing neighbor cubes for contextual data padding (works without them):")
            for cube in missing_cubes:
                print(f"{cube[0]}_{cube[1]}_{cube[2]}")
        
        return padded_raw_data

    def get_padded_data_zarr(self, zarr_arr):
        z_shape, y_shape, x_shape = zarr_arr.shape
        output_size = self.cube_config.chunk_size + 2 * self.cube_config.pad_amount
        padded_raw_data = np.zeros((output_size, output_size, output_size))

        z_start = max(0, self.cube_config.z_num - self.cube_config.pad_amount)
        z_end = min(z_shape, self.cube_config.z_num + self.cube_config.chunk_size + self.cube_config.pad_amount)
        y_start = max(0, self.cube_config.y_num - self.cube_config.pad_amount)
        y_end = min(y_shape, self.cube_config.y_num + self.cube_config.chunk_size + self.cube_config.pad_amount)
        x_start = max(0, self.cube_config.x_num - self.cube_config.pad_amount)
        x_end = min(x_shape, self.cube_config.x_num + self.cube_config.chunk_size + self.cube_config.pad_amount)

        raw_data = zarr_arr[z_start:z_end, y_start:y_end, x_start:x_end]

        z_out_start = max(0, self.cube_config.pad_amount - (self.cube_config.z_num - z_start))
        y_out_start = max(0, self.cube_config.pad_amount - (self.cube_config.y_num - y_start))
        x_out_start = max(0, self.cube_config.pad_amount - (self.cube_config.x_num - x_start))

        padded_raw_data[
            z_out_start:z_out_start + (z_end - z_start),
            y_out_start:y_out_start + (y_end - y_start),
            x_out_start:x_out_start + (x_end - x_start)
        ] = raw_data

        return padded_raw_data
    
    def combine_labeled_chunks(self, labeled_4d_array):
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

    def read_origin_file(self, file_path):
        with open(file_path, 'r') as f:
            return np.array([float(coord.strip()) for coord in f])

    def find_b2nd_files(self, root_directory):
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

    def read_b2nd_chunk(self, file_path, origin, z, y, x, chunk_size, padding):
        print(f"Processing file {file_path}")
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

    def split_connected_components(self, chunk, chunk_i):
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

    def process_b2nd_files(self, root_directory, z, y, x, chunk_size, padding):
        b2nd_files, origin_files = self.find_b2nd_files(root_directory)
        chunks = []
        chunk_i = 1

        for b2nd_file, origin_file in zip(b2nd_files, origin_files):
            origin = self.read_origin_file(origin_file)
            chunk = self.read_b2nd_chunk(b2nd_file, origin, z, y, x, chunk_size, padding)
            if chunk is not None:
                split_chunks, chunk_i = self.split_connected_components(chunk, chunk_i)
                chunks.extend(split_chunks)

        result_array = np.stack(chunks, axis=0) if chunks else np.array([])
        return self.combine_labeled_chunks(result_array)

    @staticmethod
    def threshold_mask(array_3d, factor=1.0, min_size=1000, hole_size=1000):
        sigma = 2
        array_3d = gaussian_filter(array_3d, sigma=sigma)
        
        # Flatten the 3D array to 1D for Otsu's method
        flat_array = array_3d.flatten()
        
        # Calculate Otsu's threshold
        otsu_threshold = threshold_otsu(flat_array)
        
        # Apply the factor to the Otsu threshold
        threshold = otsu_threshold / factor
        
        mask = array_3d > threshold
        
        mask = remove_small_objects(mask, min_size=min_size)
        mask = remove_small_holes(mask, area_threshold=hole_size)

        bright_spot_mask_arr = DataManager.bright_spot_mask(array_3d)
        mask = mask | bright_spot_mask_arr
        return mask

    @staticmethod
    def bright_spot_mask(data):
        threshold = np.percentile(data, 99.5)
        bright_spot_mask = (data > threshold)
        min_size = 100
        bright_spot_mask = remove_small_objects(bright_spot_mask, min_size=min_size)
        return bright_spot_mask

    @staticmethod
    def ensure_list(item):
        return item if isinstance(item, list) else [item]

    @staticmethod
    def get_transpose_params_from_shapes(shape1, shape2):
        if len(shape1) != len(shape2) or len(shape1) != 3:
            raise ValueError("Both shapes must be 3-dimensional")
        if np.prod(shape1) != np.prod(shape2):
            raise ValueError(f"Shapes must represent the same total number of elements: {shape1}, {shape2}")
        return tuple([shape2.index(s) for s in shape1])

    @staticmethod
    def get_transpose_params_from_axis_order(source_order, target_order):
        if len(source_order) != 3 or len(target_order) != 3:
            raise ValueError("Both orders must be 3-dimensional (e.g., 'zyx')")
        
        if set(source_order) != set(target_order):
            raise ValueError("Both orders must contain the same axes (x, y, and z)")
        
        source_indices = {axis: index for index, axis in enumerate(source_order)}
        return tuple(source_indices[axis] for axis in target_order)

    @staticmethod
    def bright_spot_mask_dask(data):
        flat_data = data.flatten()
        threshold = np.percentile(flat_data, 99.5)
        mask = data > threshold
        return mask

    async def save_label_data_async(self, z,y,x, data, label_type):
        print(f"Saving label data for {z}_{y}_{x}")
        self.is_saving = True
        try:
            file_path = self.get_label_file_path(z,y,x, label_type)
            header = DataManager.create_default_nrrd_header(data, z, y, x, 'gzip')
            await self._save_nrrd_async(file_path, data, header)
        finally: 
            self.is_saving = False
    
    async def save_seg_mesh_data_async(self, z, y, x, data):
        file_path = self.get_seg_mesh_file_path(z, y, x)
        header = DataManager.create_default_nrrd_header(data, z, y, x)
        await self._save_nrrd_async(file_path, data, header)
        print(f"Saved segmentation mesh to {file_path}")

    async def save_raw_data_async(self, z, y, x):
        file_path = self.get_raw_data_file_path(z, y, x)
        
        if not os.path.exists(file_path):
            print("saving raw data at path: ", file_path)
            header = DataManager.create_default_nrrd_header(self.raw_data, z, y, x)
            await self._save_nrrd_async(file_path, self.raw_data, header)
        else:
            print(f"Raw data nrrd already exists at {file_path}")

    async def _save_nrrd_async(self, file_path, data, header=None):
        def save_task():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            nrrd.write(file_path, data, header=header)
            print(f"Saved: {file_path}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_task)

    def get_label_file_path(self,z,y,x, label_type):
        return os.path.join(self.get_output_path(z,y,x), 
                            f"{z}_{y}_{x}_zyx_{self.cube_config.chunk_size}_chunk_{self.cube_config.scroll_name}_{label_type}_label.nrrd")

    def get_raw_data_file_path(self,z,y,x,):
        return os.path.join(self.get_output_path(z,y,x), 
                            f"{z}_{y}_{x}_zyx_{self.cube_config.chunk_size}_chunk_{self.cube_config.scroll_name}_vol_raw.nrrd")
    
    def get_seg_mesh_file_path(self, z, y, x):
        return os.path.join(self.get_output_path(z, y, x), 
                            f"{z}_{y}_{x}_zyx_{self.cube_config.chunk_size}_chunk_{self.cube_config.scroll_name}_seg_mesh.nrrd")

    def get_output_path(self, z,y,x):
        return os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.cube_config.scroll_name}', 
                            f'{z}_{y}_{x}')
    
    def create_default_nrrd_header(data,z=0,y=0,x=0,encoding='raw'):
        header = {
            'type': data.dtype.name,
            'dimension': data.ndim,
            'space': 'left-posterior-superior',
            'sizes': data.shape,
            'space directions': np.eye(data.ndim).tolist(),
            'kinds': ['domain'] * data.ndim,
            'endian': 'little',
            'encoding': encoding,
            'space origin': [float(z), float(y), float(x)]  # Use a list of floats
        }
        return header