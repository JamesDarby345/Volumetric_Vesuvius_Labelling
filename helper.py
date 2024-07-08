import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage.morphology import dilation, square, remove_small_objects
import random
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
import os
import nrrd
from qtpy.QtWidgets import QMessageBox
from napari.utils.colormaps import DirectLabelColormap
from collections import defaultdict
from vispy.scene.cameras.perspective import PerspectiveCamera
from vispy.util import keys
from collections import deque
import numba
import zarr 
import ast

def ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            # Try to parse the string as a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If it's not a valid list representation, wrap it in a list
            return [value]
    return [value] if value is not None else []

#monkey-patch the camera controls
def patched_viewbox_mouse_event(self, event):
    if event.handled or not self.interactive:
        return
    try:
        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.type == 'mouse_release':
            self._event_value = None  # Reset
        elif event.type == 'mouse_press':
            self._event_value = event.pos[:2]  # Only take the first two values
            event.handled = True
        elif event.type == 'mouse_move':
            modifiers = event.mouse_event.modifiers
            if event.press_event is None:
                return
            if 1 in event.buttons and 2 in event.buttons:
                return
            #temp
            # if 2 in event.buttons and keys.SHIFT in modifiers:
            #     return

            p1 = event.mouse_event.press_event.pos[:2] # Only take the first two values
            p2 = event.mouse_event.pos[:2] # Only take the first two values
            d = p2 - p1

            if (1 in event.buttons and not modifiers) or (2 in event.buttons and not modifiers):
                # Rotate
                self._update_rotation(event)

            # elif 2 in event.buttons and not modifiers:
            #     # Zoom
            #     if self._event_value is None:
            #         self._event_value = (self._scale_factor, self._distance)
            #     zoomy = (1 + self.zoom_factor) ** d[1]

            #     self.scale_factor = self._event_value[0] * zoomy
            #     # Modify distance if its given
            #     if self._distance is not None:
            #         self._distance = self._event_value[1] * zoomy
            #     self.view_changed()

            # This is the modified condition
            elif (1 in event.buttons and keys.SHIFT in modifiers) or 3 in event.buttons or (2 in event.buttons and keys.SHIFT in modifiers):
                # Translate
                norm = np.mean(self._viewbox.size)
                if self._event_value is None or len(self._event_value) == 2:
                    self._event_value = self.center
                dist = (p1 - p2) / norm * self._scale_factor
                dist[1] *= -1
                # Black magic part 1: turn 2D into 3D translations
                dx, dy, dz = self._dist_to_trans(dist)
                # Black magic part 2: take up-vector and flipping into account
                ff = self._flip_factors
                up, forward, right = self._get_dim_vectors()
                dx, dy, dz = right * dx + forward * dy + up * dz
                dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
                c = self._event_value
                self.center = c[0] + dx, c[1] + dy, c[2] + dz

            # elif 2 in event.buttons and keys.SHIFT in modifiers:
            #     p1 = event.mouse_event.press_event.pos 
            #     p2 = event.mouse_event.pos 
            #     d = p2 - p1
            #     # Change fov
            #     print("abcd", self._event_value)
            #     print(self._fov)
            #     print(d[1])
            #     # exit()
            #     if self._event_value is None:
            #         self._event_value = self._fov
            #     fov = self._event_value - d[1] / 5.0
            #     self.fov = min(180.0, max(0.0, fov))
    except AttributeError as e:
        print(f"AttributeError in patched_viewbox_mouse_event: {e}")
    except ValueError as e:
        print(f"ValueError in patched_viewbox_mouse_event: {e}")
    except TypeError as e:
        print(f"TypeError in patched_viewbox_mouse_event: {e}")
    except Exception as e:
        print(f"Unexpected error in patched_viewbox_mouse_event: {e}")

@numba.jit(nopython=True, parallel=True)
def numba_dilation_3d_labels(data, iterations):
    result = data.copy()
    rows, cols, depths = data.shape
    
    for _ in range(iterations):
        temp = result.copy()
        for i in numba.prange(rows):
            for j in range(cols):
                for k in range(depths):
                    if result[i, j, k] == 0:  # Only dilate into empty space
                        # Check 6-connected neighbors
                        if i > 0 and temp[i-1, j, k] != 0:
                            result[i, j, k] = temp[i-1, j, k]
                        elif i < rows-1 and temp[i+1, j, k] != 0:
                            result[i, j, k] = temp[i+1, j, k]
                        elif j > 0 and temp[i, j-1, k] != 0:
                            result[i, j, k] = temp[i, j-1, k]
                        elif j < cols-1 and temp[i, j+1, k] != 0:
                            result[i, j, k] = temp[i, j+1, k]
                        elif k > 0 and temp[i, j, k-1] != 0:
                            result[i, j, k] = temp[i, j, k-1]
                        elif k < depths-1 and temp[i, j, k+1] != 0:
                            result[i, j, k] = temp[i, j, k+1]
                        
    return result

def show_popup(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle("Popup")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()

def confirm_popup(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setText(message)
    msg.setWindowTitle("Confirmation")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    return msg.exec_()

def get_padded_data_zarr(zarr_arr, z_num, y_num, x_num, chunk_size, pad_amount):
    # Get the shape of the zarr array
    z_shape, y_shape, x_shape = zarr_arr.shape

    # Initialize the output array with zeros
    output_size = chunk_size + 2 * pad_amount
    padded_raw_data = np.zeros((output_size, output_size, output_size))

    # Compute indices for the zarr array
    z_start = max(0, z_num - pad_amount)
    z_end = min(z_shape, z_num + chunk_size + pad_amount)
    y_start = max(0, y_num - pad_amount)
    y_end = min(y_shape, y_num + chunk_size + pad_amount)
    x_start = max(0, x_num - pad_amount)
    x_end = min(x_shape, x_num + chunk_size + pad_amount)

    # Get data from zarr array
    raw_data = zarr_arr[z_start:z_end, y_start:y_end, x_start:x_end]

    # Compute indices for placing data in the output array
    z_out_start = max(0, pad_amount - (z_num - z_start))
    y_out_start = max(0, pad_amount - (y_num - y_start))
    x_out_start = max(0, pad_amount - (x_num - x_start))

    # Place the data in the output array
    padded_raw_data[
        z_out_start:z_out_start + (z_end - z_start),
        y_out_start:y_out_start + (y_end - y_start),
        x_out_start:x_out_start + (x_end - x_start)
    ] = raw_data

    return padded_raw_data

def get_padded_nrrd_data(folder_path, z, y, x, pad_amount, chunk_size=256):
    padded_raw_data = None
    missing_cubes = []
    
    # Define the relative coordinates of neighboring chunks
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
    
    padded_size = chunk_size + 2 * pad_amount
    padded_raw_data = np.zeros((padded_size, padded_size, padded_size))
    
    for dz, dy, dx in neighbors:
        neighbor_z = str(int(z) + dz * chunk_size).zfill(5)
        neighbor_y = str(int(y) + dy * chunk_size).zfill(5)
        neighbor_x = str(int(x) + dx * chunk_size).zfill(5)
        
        filename = f"{neighbor_z}_{neighbor_y}_{neighbor_x}/{neighbor_z}_{neighbor_y}_{neighbor_x}_volume.nrrd"
        filepath = os.path.join(folder_path, filename)
        
        if os.path.exists(filepath):
            try:
                data, header = nrrd.read(filepath)
            except StopIteration:
                print(f"Error: Unable to read the NRRD file header from {filepath}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
            
            # Determine the slices to extract from this cube
            z_start = chunk_size - pad_amount if dz < 0 else 0
            z_end = pad_amount if dz > 0 else chunk_size
            y_start = chunk_size - pad_amount if dy < 0 else 0
            y_end = pad_amount if dy > 0 else chunk_size
            x_start = chunk_size - pad_amount if dx < 0 else 0
            x_end = pad_amount if dx > 0 else chunk_size

            # print(dz, dy, dx)
            # print(f"Extracting data from {filename} at zyx {z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}")
            
            # Extract the relevant portion of the data
            extracted_data = data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Determine where to place the extracted data in padded_raw_data
            z_pad_start = pad_amount + (dz) * pad_amount
            if dz == 1:
                z_pad_start = pad_amount + chunk_size
            y_pad_start = pad_amount + (dy) * pad_amount
            if dy == 1:
                y_pad_start = pad_amount + chunk_size
            x_pad_start = pad_amount + (dx) * pad_amount
            if dx == 1:
                x_pad_start = pad_amount + chunk_size
            
            z_pad_end = z_pad_start + extracted_data.shape[0]
            y_pad_end = y_pad_start + extracted_data.shape[1]
            x_pad_end = x_pad_start + extracted_data.shape[2]
            
            # print(f"Placing data in padded_raw_data at zyx {z_pad_start}:{z_pad_end}, {y_pad_start}:{y_pad_end}, {x_pad_start}:{x_pad_end}")


            # Place the extracted data in padded_raw_data
            padded_raw_data[z_pad_start:z_pad_end, y_pad_start:y_pad_end, x_pad_start:x_pad_end] = extracted_data
        else:
            missing_cubes.append((neighbor_z, neighbor_y, neighbor_x))
    
    if len(missing_cubes) > 0:
        print("List of missing neighbor cubes for contextual data padding (works without them):")
        for cube in missing_cubes:
            print(f"{cube[0]}_{cube[1]}_{cube[2]}")
    
    return padded_raw_data

def limited_bfs_flood_fill(data, start_coords, max_distance):
    shape = data.shape
    filled = np.zeros(shape, dtype=bool)
    value = data[start_coords]
    
    # Directions for 6-connectivity in 3D (x, y, z)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # BFS queue
    queue = deque([(start_coords, 0)])  # (coordinates, distance)
    filled[start_coords] = True

    while queue:
        (z, y, x), dist = queue.popleft()
        
        if dist < max_distance:
            for dz, dy, dx in directions:
                nz, ny, nx = z + dz, y + dy, x + dx
                
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                    if not filled[nz, ny, nx] and data[nz, ny, nx] == value:
                        filled[nz, ny, nx] = True
                        queue.append(((nz, ny, nx), dist + 1))

    return filled

def label_foreground_structures_napari(input_array, min_size=1000):
    # Find connected components in the foreground (value 2)
    foreground = input_array > 0

    # Label connected components
    labeled_array, num_features = scipy.ndimage.label(foreground)
    
    # Measure the size of each connected component
    structure_sizes = np.array(scipy.ndimage.sum(foreground, labeled_array, range(num_features + 1)))
    
    # Create a mask to remove small structures
    remove_mask = structure_sizes < min_size
    remove_mask[0] = 0  # Ensure the background is not removed

    # Remove small structures
    labeled_array[remove_mask[labeled_array]] = 0

    # Relabel the structures after removal
    labeled_array, num_features = scipy.ndimage.label(labeled_array > 0)

    print(f"Number of connected foreground structures before filtering: {num_features}")
    print(f"Number of connected foreground structures after filtering: {np.max(labeled_array)}")

    return labeled_array

def bright_spot_mask(data):
    # Calculate the threshold for the top 0.1% brightest voxels
    threshold = np.percentile(data, 99.5)

    # Create a mask for the top 1% brightest voxels
    bright_spot_mask = (data > threshold)

    # Apply small object removal (you can adjust the minimum size as needed)
    min_size = 100  # Minimum size of objects to keep
    bright_spot_mask = remove_small_objects(bright_spot_mask, min_size=min_size)

    # Apply small hole removal (you can adjust the area threshold as needed)
    # area_threshold = 20  # Maximum area of holes to fill
    # bright_spot_mask = remove_small_holes(bright_spot_mask, area_threshold=area_threshold)
    # Dilate the bright spot mask by one

    bright_spot_mask = binary_dilation(bright_spot_mask, iterations=1)
    return bright_spot_mask


def label_foreground_structures(input_array, min_size=1000, foreground_value=2):
    """
    Label connected foreground structures in the input array, removing small structures below a specified size.
    
    Parameters:
        input_array (np.ndarray): The input array with foreground structures labeled as 2.
        min_size (int): Minimum size of the structures to retain. Structures smaller than this size will be removed.
    
    Returns:
        np.ndarray: The labeled array with small structures removed and remaining structures relabeled.
    """
    
    # Find connected components in the foreground (value 2)
    foreground = input_array == foreground_value
    
    # Label connected components
    labeled_array, num_features = scipy.ndimage.label(foreground)
    
    # Measure the size of each connected component
    structure_sizes = np.array(scipy.ndimage.sum(foreground, labeled_array, range(num_features + 1)))
    
    # Create a mask to remove small structures
    remove_mask = structure_sizes < min_size
    remove_mask[0] = 0  # Ensure the background is not removed

    # Remove small structures
    labeled_array[remove_mask[labeled_array]] = 0

    # Relabel the structures after removal
    labeled_array, num_features = scipy.ndimage.label(labeled_array > 0)

    print(f"Number of connected foreground structures before filtering: {num_features}")
    print(f"Number of connected foreground structures after filtering: {np.max(labeled_array)}")
    
    return labeled_array

def mark_boundaries_color(image, label_img, color=None, outline_color=None, mode='outer', background_label=0, dilation_size=1):
    """Return image with boundaries between labeled regions highlighted with consistent colors derived from labels.

    Parameters:
    - image: Input image.
    - label_img: Image with labeled regions.
    - color: Ignored in this version.
    - outline_color: If specified, use this color for the outline. Otherwise, use the same as boundary.
    - mode: Choose 'inner', 'outer', or 'thick' to define boundary type.
    - background_label: Label to be treated as the background.
    - dilation_size: Size of the dilation square for the boundaries.

    Returns:
    - Image with boundaries highlighted.
    """
    # Ensure input image is in float and has three channels
    float_dtype = np.float32  # Use float32 for efficiency
    marked = img_as_float(image, force_copy=True).astype(float_dtype, copy=False)
    if marked.ndim == 2:
        marked = gray2rgb(marked)

    # Create a color map normalized by the number of unique labels
    unique_labels = np.unique(label_img)
    color_map = plt.get_cmap('nipy_spectral')  # You can change 'nipy_spectral' to any other colormap

    # Find boundaries and apply colors
    boundaries = find_boundaries(label_img, mode=mode, background=background_label)
    for label in unique_labels:
        if label == background_label:
            continue
        # Normalize label value to the range of the colormap
        normalized_color = color_map(label / np.max(unique_labels))[:3]  # Get RGB values only
        label_boundaries = find_boundaries(label_img == label, mode=mode)
        label_boundaries = dilation(label_boundaries, square(dilation_size))
        marked[label_boundaries] = normalized_color
        if outline_color is not None:
            outlines = dilation(label_boundaries, square(dilation_size + 1))
            marked[outlines] = outline_color
        else:
            marked[label_boundaries] = normalized_color

    return marked


def consistent_color(label):
    """Generate a consistent color for a given label using a hash function."""
    random.seed(hash(label))
    return [random.random() for _ in range(3)]

def mark_boundaries_multicolor(image, label_img, color=None, outline_color=None, mode='outer', background_label=0, dilation_size=1):
    """Return image with boundaries between labeled regions highlighted with consistent colors.

    Parameters are the same as in the original function but color is ignored if provided.
    """
    # Ensure input image is in float and has three channels
    float_dtype = np.float32  # Use float32 for efficiency
    marked = img_as_float(image, force_copy=True).astype(float_dtype, copy=False)
    if marked.ndim == 2:
        marked = gray2rgb(marked)

    # Generate consistent colors for each unique label in label_img
    unique_labels = np.unique(label_img)
    color_map = {label: consistent_color(label) for label in unique_labels if label != background_label}

    # Find boundaries and apply colors
    boundaries = find_boundaries(label_img, mode=mode, background=background_label)
    for label, color in color_map.items():
        label_boundaries = find_boundaries(label_img == label, mode=mode)
        label_boundaries = dilation(label_boundaries, square(dilation_size))
        if outline_color is not None:
            outlines = dilation(label_boundaries, square(dilation_size))
            marked[outlines] = outline_color
        marked[label_boundaries] = color

    return marked

def plot_segmentation_results(test_slice, segmentation):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show marked boundary image
    axes[0].imshow(mark_boundaries(test_slice, np.array(segmentation)))
    axes[0].set_title("Marked Boundary")

    # Show unmarked boundary image
    axes[1].imshow(test_slice, cmap='gray')
    axes[1].set_title("Unmarked Boundary")

    plt.show()

def get_slicer_colormap():
    return {
        0: [0, 0, 0, 0],
        1: [128, 174, 128, 255],
        2: [241, 214, 145, 255],
        3: [177, 122, 101, 255],
        4: [111, 184, 210, 255],
        5: [216, 101, 79, 255],
        6: [221, 130, 101, 255],
        7: [144, 238, 144, 255],
        8: [192, 104, 88, 255],
        9: [220, 245, 20, 255],
        10: [78, 63, 0, 255],
        11: [255, 250, 220, 255],
        12: [230, 220, 70, 255],
        13: [200, 200, 235, 255],
        14: [250, 250, 210, 255],
        15: [244, 214, 49, 255],
        16: [0, 151, 206, 255],
        17: [216, 101, 79, 255],
        18: [183, 156, 220, 255],
        19: [183, 214, 211, 255],
        20: [152, 189, 207, 255],
        21: [111, 184, 210, 255],
        22: [178, 212, 242, 255],
        23: [68, 172, 100, 255],
        24: [111, 197, 131, 255],
        25: [85, 188, 255, 255],
        26: [0, 145, 30, 255],
        27: [214, 230, 130, 255],
        28: [78, 63, 0, 255],
        29: [218, 255, 255, 255],
        30: [170, 250, 250, 255],
        31: [140, 224, 228, 255],
        32: [188, 65, 28, 255],
        33: [216, 191, 216, 255],
        34: [145, 60, 66, 255],
        35: [150, 98, 83, 255],
        36: [177, 122, 101, 255],
        37: [244, 214, 49, 255],
        38: [250, 250, 225, 255],
        39: [200, 200, 215, 255],
        40: [68, 131, 98, 255],
    }

def get_direct_label_colormap():
    slicer_colormap = get_slicer_colormap()
    
    # Normalize colors to 0-1 range
    normalized_colormap = defaultdict(lambda: np.array([0, 0, 0, 0]))
    for k, color in slicer_colormap.items():
        normalized_colormap[k] = np.array(color) / 255
    
    # Add None key with a default color (e.g., transparent black)
    normalized_colormap[None] = np.array([0, 0, 0, 0])
    
    # Create the DirectLabelColormap
    return DirectLabelColormap(color_dict=normalized_colormap)