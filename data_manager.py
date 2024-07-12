# data_manager.py

import os
import nrrd
import numpy as np
import zarr
from dask import array as da
from datetime import datetime
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, remove_small_holes
import asyncio
import ast

class DataManager:
    def __init__(self, config):
        self.config = config
        self.raw_data = None
        self.padded_raw_data = None
        self.original_label_data = None
        self.original_ink_pred_data = None
        self.label_header = None
        self.raw_data_zarr_shape = None
        self.nrrd_cube_path = self.config.nrrd_cube_path

        self.load_data()

    def reload_data(self, z, y, x):
        print(f"Reloading data in data manager {self.config.z}_{self.config.y}_{self.config.x}",z,y,x)
        self.config.update_coordinates(z=z, y=y, x=x)
        print(f"Reloading data in data manager {self.config.z}_{self.config.y}_{self.config.x}")
        # Reset all data attributes
        self.raw_data = None
        self.padded_raw_data = None
        self.original_label_data = None
        self.original_ink_pred_data = None
        self.label_header = None
        self.raw_data_zarr_shape = None
        self.nrrd_cube_path = self.config.nrrd_cube_path

        # Reload all data
        self.load_data()

    def load_data(self):
        self.load_raw_data()
        self.load_label_data()
        self.load_ink_pred_data()
            

    def load_raw_data(self):
        if not self.config.using_raw_data_zarr:
            volume_file_path = os.path.join(self.config.nrrd_cube_path, 
                                            f'{self.config.z}_{self.config.y}_{self.config.x}', 
                                            f'{self.config.z}_{self.config.y}_{self.config.x}_volume.nrrd')
            self.raw_data, _ = nrrd.read(volume_file_path)
            self.padded_raw_data = self.get_padded_nrrd_data()
        else:
            raw_data_zarr_multi_res = zarr.open(self.config.raw_data_zarr_path, mode='r')
            self.raw_data_zarr_shape = raw_data_zarr_multi_res[0].shape
            self.raw_data = raw_data_zarr_multi_res[0][self.config.z_num:self.config.z_num+self.config.chunk_size, 
                                                       self.config.y_num:self.config.y_num+self.config.chunk_size, 
                                                       self.config.x_num:self.config.x_num+self.config.chunk_size]
            self.padded_raw_data = self.get_padded_data_zarr(raw_data_zarr_multi_res[0])

    def load_label_data(self):
        output_folder_path = os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.config.scroll_name}')
        saved_label_file_path = os.path.join(output_folder_path, 
                                             f"{self.config.z}_{self.config.y}_{self.config.x}",
                                             f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_vol_label.nrrd")

        nrrd_cube_folder_path = os.path.join(self.config.nrrd_cube_path, f'{self.config.z}_{self.config.y}_{self.config.x}')
        mask_file_path = os.path.join(nrrd_cube_folder_path, f'{self.config.z}_{self.config.y}_{self.config.x}_mask.nrrd')

        if os.path.exists(saved_label_file_path):
            self.original_label_data, self.label_header = nrrd.read(saved_label_file_path)
        elif os.path.exists(mask_file_path):
            self.original_label_data, self.label_header = nrrd.read(mask_file_path)
        else:
            self.original_label_data = self.threshold_mask(self.raw_data, factor=self.config.factor).astype(np.uint8)
            os.makedirs(nrrd_cube_folder_path, exist_ok=True)
            print("Creating Papyrus Label from thresholded raw data, may take a few seconds...")
            nrrd.write(mask_file_path, self.original_label_data)
            self.original_label_data, self.label_header = nrrd.read(mask_file_path)

        if self.label_header is not None:
            print(self.label_header)
            self.label_header = defaultdict(list, self.label_header)
            for key in ['saved_timestamps', 'open_timestamps']:
                self.label_header[key] = self.ensure_list(self.label_header[key])
            self.label_header['open_timestamps'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if 'author' not in self.label_header and self.config.author is not None and self.config.author != '':
                self.label_header['author'] = self.config.author

    def load_ink_pred_data(self):
        saved_ink_pred_file_path = os.path.join(os.getcwd(), 'output', 
                                                f'volumetric_labels_{self.config.scroll_name}', 
                                                f"{self.config.z}_{self.config.y}_{self.config.x}",
                                                f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_ink_label.nrrd")

        if os.path.exists(saved_ink_pred_file_path):
            self.original_ink_pred_data, _ = nrrd.read(saved_ink_pred_file_path)
        elif self.config.using_ink_pred_zarr:
            ink_pred_zarr = zarr.open(self.config.ink_pred_zarr_path, mode='r')
            ink_pred_dask = da.from_zarr(ink_pred_zarr)

            if self.config.raw_data_axis_order is None or self.config.ink_pred_label_axis_order is None and self.raw_data_zarr_shape is not None:
                transpose_params = self.get_transpose_params_from_shapes(self.raw_data_zarr_shape, ink_pred_dask.shape) 
                ink_pred_dask = ink_pred_dask.transpose(transpose_params)
            elif self.config.raw_data_axis_order is not None and self.config.ink_pred_label_axis_order is not None:
                transpose_params = self.get_transpose_params_from_axis_order(self.config.raw_data_axis_order, self.config.ink_pred_label_axis_order)
                ink_pred_dask = ink_pred_dask.transpose(transpose_params)
            else:
                print("Could not ensure alignment of ink prediction label with raw data, using default axis order. Please check the axis order of the raw data and ink prediction label and set it in the config file if incorrect")
            
            ink_pred_region = ink_pred_dask[self.config.z_num:self.config.z_num+self.config.chunk_size, 
                                            self.config.y_num:self.config.y_num+self.config.chunk_size, 
                                            self.config.x_num:self.config.x_num+self.config.chunk_size]
            thresholded_data = np.array(da.where(ink_pred_region < self.config.ink_threshold, 0, 1)).astype(np.uint8)
            self.original_ink_pred_data = thresholded_data

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
        
        padded_size = self.config.chunk_size + 2 * self.config.pad_amount
        padded_raw_data = np.zeros((padded_size, padded_size, padded_size))
        
        for dz, dy, dx in neighbors:
            neighbor_z = str(int(self.config.z) + dz * self.config.chunk_size).zfill(5)
            neighbor_y = str(int(self.config.y) + dy * self.config.chunk_size).zfill(5)
            neighbor_x = str(int(self.config.x) + dx * self.config.chunk_size).zfill(5)
            
            filename = f"{neighbor_z}_{neighbor_y}_{neighbor_x}/{neighbor_z}_{neighbor_y}_{neighbor_x}_volume.nrrd"
            filepath = os.path.join(self.config.nrrd_cube_path, filename)
            
            if os.path.exists(filepath):
                try:
                    data, header = nrrd.read(filepath)
                except Exception as e:
                    print(f"An error occurred while reading {filepath}: {e}")
                    continue
                
                z_start = self.config.chunk_size - self.config.pad_amount if dz < 0 else 0
                z_end = self.config.pad_amount if dz > 0 else self.config.chunk_size
                y_start = self.config.chunk_size - self.config.pad_amount if dy < 0 else 0
                y_end = self.config.pad_amount if dy > 0 else self.config.chunk_size
                x_start = self.config.chunk_size - self.config.pad_amount if dx < 0 else 0
                x_end = self.config.pad_amount if dx > 0 else self.config.chunk_size

                extracted_data = data[z_start:z_end, y_start:y_end, x_start:x_end]
                
                z_pad_start = self.config.pad_amount + (dz) * self.config.pad_amount
                if dz == 1:
                    z_pad_start = self.config.pad_amount + self.config.chunk_size
                y_pad_start = self.config.pad_amount + (dy) * self.config.pad_amount
                if dy == 1:
                    y_pad_start = self.config.pad_amount + self.config.chunk_size
                x_pad_start = self.config.pad_amount + (dx) * self.config.pad_amount
                if dx == 1:
                    x_pad_start = self.config.pad_amount + self.config.chunk_size
                
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
        output_size = self.config.chunk_size + 2 * self.config.pad_amount
        padded_raw_data = np.zeros((output_size, output_size, output_size))

        z_start = max(0, self.config.z_num - self.config.pad_amount)
        z_end = min(z_shape, self.config.z_num + self.config.chunk_size + self.config.pad_amount)
        y_start = max(0, self.config.y_num - self.config.pad_amount)
        y_end = min(y_shape, self.config.y_num + self.config.chunk_size + self.config.pad_amount)
        x_start = max(0, self.config.x_num - self.config.pad_amount)
        x_end = min(x_shape, self.config.x_num + self.config.chunk_size + self.config.pad_amount)

        raw_data = zarr_arr[z_start:z_end, y_start:y_end, x_start:x_end]

        z_out_start = max(0, self.config.pad_amount - (self.config.z_num - z_start))
        y_out_start = max(0, self.config.pad_amount - (self.config.y_num - y_start))
        x_out_start = max(0, self.config.pad_amount - (self.config.x_num - x_start))

        padded_raw_data[
            z_out_start:z_out_start + (z_end - z_start),
            y_out_start:y_out_start + (y_end - y_start),
            x_out_start:x_out_start + (x_end - x_start)
        ] = raw_data

        return padded_raw_data

    @staticmethod
    def threshold_mask(array_3d, factor=1.0, min_size=1000, hole_size=1000):
        sigma = 2
        array_3d = gaussian_filter(array_3d, sigma=sigma)
        threshold = np.mean(array_3d) / factor
        
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
        # if isinstance(value, list):
        #     return value
        # if isinstance(value, str):
        #     try:
        #         # Try to parse the string as a list
        #         return ast.literal_eval(value)
        #     except (ValueError, SyntaxError):
        #         # If it's not a valid list representation, wrap it in a list
        #         return [value]
        # return [value] if value is not None else []

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

    @staticmethod
    def is_valid_coord_s1(num_or_list):
        if isinstance(num_or_list, (int, float)):
            difference = abs(num_or_list - 2000)
            return difference % 256 == 0
        elif isinstance(num_or_list, (list, np.ndarray)):
            differences = np.abs(np.array(num_or_list) - 2000)
            return (differences % 256 == 0).all()
        else:
            raise TypeError("Input must be a number, list, or numpy array")

    @staticmethod
    def find_nearest_valid_coord(num):
        difference = num - 2000
        quotient = difference // 256
        remainder = difference % 256

        if remainder == 0:
            result = num
        elif remainder > 128:
            result = 2000 + 256 * (quotient + 1)
        else:
            result = 2000 + 256 * quotient
        
        return max(result, 208)

    def save_label_data(self, label_data, label_type='vol'):
        output_folder_path = os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.config.scroll_name}')
        file_path = os.path.join(output_folder_path, 
                                 f"{self.config.z}_{self.config.y}_{self.config.x}",
                                 f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_{label_type}_label.nrrd")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if label_type == 'vol' and self.label_header is not None:
            self.label_header['saved_timestamps'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            nrrd.write(file_path, label_data, header=dict(self.label_header))
        else:
            nrrd.write(file_path, label_data)
        
        print(f"Saved {label_type} label to {file_path}")

    def save_raw_data(self):
        output_folder_path = os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.config.scroll_name}')
        file_path = os.path.join(output_folder_path, 
                                 f"{self.config.z}_{self.config.y}_{self.config.x}",
                                 f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_vol_raw.nrrd")
        
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            nrrd.write(file_path, self.raw_data)
            print(f"Saved raw data to {file_path}")
        else:
            print(f"Raw data file already exists at {file_path}")

    async def save_label_data_async(self, data, label_type):
        file_path = self.get_label_file_path(label_type)
        
        if label_type == 'vol':
            header = self.label_header
            if header is not None:
                # Parse the existing timestamps
                if 'saved_timestamps' in header:
                    try:
                        saved_timestamps = ast.literal_eval(header['saved_timestamps'])
                    except:
                        saved_timestamps = []
                else:
                    saved_timestamps = []
                
                # Append the new timestamp
                saved_timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                # Update the header with the new list of timestamps
                header['saved_timestamps'] = str(saved_timestamps)
                
                # Do the same for open_timestamps
                if 'open_timestamps' in header:
                    try:
                        open_timestamps = ast.literal_eval(header['open_timestamps'])
                    except:
                        open_timestamps = []
                else:
                    open_timestamps = []
                header['open_timestamps'] = str(open_timestamps)
                
                header = dict(header)
        else:
            header = None

        await self._save_nrrd_async(file_path, data, header)

    async def save_raw_data_async(self):
        file_path = self.get_raw_data_file_path()
        if not os.path.exists(file_path):
            await self._save_nrrd_async(file_path, self.raw_data)

    async def _save_nrrd_async(self, file_path, data, header=None):
        def save_task():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            nrrd.write(file_path, data, header=header)
            print(f"Saved: {file_path}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_task)

    def get_label_file_path(self, label_type):
        return os.path.join(self.get_output_path(), 
                            f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_{label_type}_label.nrrd")

    def get_raw_data_file_path(self):
        return os.path.join(self.get_output_path(), 
                            f"{self.config.z}_{self.config.y}_{self.config.x}_zyx_{self.config.chunk_size}_chunk_{self.config.scroll_name}_vol_raw.nrrd")

    def get_output_path(self):
        return os.path.join(os.getcwd(), 'output', f'volumetric_labels_{self.config.scroll_name}', 
                            f'{self.config.z}_{self.config.y}_{self.config.x}')