import numpy as np
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from skimage.morphology import dilation, square, remove_small_objects, remove_small_holes
import random
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
import os
import nrrd
from qtpy.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QInputDialog, QWidget
from napari.utils.colormaps import DirectLabelColormap
from collections import defaultdict
from vispy.scene.cameras.perspective import PerspectiveCamera
from vispy.util import keys
from collections import deque
import numba
import zarr 
import ast
from sklearn.decomposition import PCA
from skimage import measure


def get_transpose_params_from_shapes(shape1, shape2):
    # Check if the shapes have the same number of dimensions
    if len(shape1) != len(shape2) or len(shape1) != 3:
        raise ValueError("Both shapes must be 3-dimensional")

    # Check if the shapes have the same total number of elements
    if np.prod(shape1) != np.prod(shape2):
        raise ValueError(f"Shapes must represent the same total number of elements: {shape1}, {shape2}")

    # Find the permutation of axes
    permutation = [shape2.index(s) for s in shape1]

    return tuple(permutation)

def get_transpose_params_from_axis_order(source_order, target_order):
    if len(source_order) != 3 or len(target_order) != 3:
        raise ValueError("Both orders must be 3-dimensional (e.g., 'zyx')")
    
    if set(source_order) != set(target_order):
        raise ValueError("Both orders must contain the same axes (x, y, and z)")
    
    # Create a dictionary mapping each axis in the source to its index
    source_indices = {axis: index for index, axis in enumerate(source_order)}
    
    # Create the transpose parameters by looking up each target axis in the source
    transpose_params = tuple(source_indices[axis] for axis in target_order)
    
    return transpose_params

def find_best_intersecting_plane_napari(array_3d):
    # Convert 3D array to point cloud
    points = np.array(np.where(array_3d != 0)).T

    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(points)

    # The normal vector of the plane is the third principal component
    normal_vector = pca.components_[2]

    # Ensure the normal vector points "up" (positive z direction)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # The point on the plane is the mean of all points
    point_on_plane = np.mean(points, axis=0)

    # Convert to Napari coordinates (x, y, z)
    napari_position = point_on_plane
    napari_normal = normal_vector

    return napari_position, napari_normal

def bright_spot_mask_dask(data):
    # Flatten the multi-dimensional Dask array
    flat_data = data.flatten()
    
    # Compute the 99.5 percentile on the flattened array
    threshold = np.percentile(flat_data, 99.5)
    
    # Create a mask based on the threshold
    mask = data > threshold
    
    return mask

def is_valid_coord(num_or_list):
    if isinstance(num_or_list, (int, float)):
        difference = abs(num_or_list - 2000)
        return difference % 256 == 0
    elif isinstance(num_or_list, (list, np.ndarray)):
        differences = np.abs(np.array(num_or_list) - 2000)
        return (differences % 256 == 0).all()
    else:
        raise TypeError("Input must be a number, list, or numpy array")

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
    
    # Ensure the result is always greater than 0
    return max(result, 208)


def threshold_mask(array_3d, factor=1.0, min_size=200, hole_size=200):
    # Calculate the mean of the entire 3D array
    threshold = np.mean(array_3d) / factor
    
    # Create initial mask
    mask = array_3d > threshold
    
    # Remove small objects and holes
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=hole_size)

    #remove bright spots, top 0.5% brightest voxels
    bright_spot_mask_arr = bright_spot_mask(array_3d)
    mask = mask | bright_spot_mask_arr
    return mask

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

def select_from_list_popup(title, message, options):
    parent = QWidget()  # You might want to pass a proper parent widget if available
    selected_item, ok = QInputDialog.getItem(parent, title, message, options, 0, False)
    
    if ok and selected_item:
        return selected_item
    else:
        return None

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

# def label_foreground_structures_napari(input_array, min_size=10, chunk_size=200):
#     foreground = input_array > 0
#     shape = foreground.shape
#     labeled_array = np.zeros(shape, dtype=np.int32)
#     max_label = 0

#     for z in range(0, shape[0], chunk_size):
#         for y in range(0, shape[1], chunk_size):
#             for x in range(0, shape[2], chunk_size):
#                 chunk = foreground[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
#                 chunk_labeled = measure.label(chunk, connectivity=3)
#                 chunk_labeled[chunk_labeled > 0] += max_label
#                 labeled_array[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size] = chunk_labeled
#                 max_label = labeled_array.max()

#     # Measure the size of each connected component
#     structure_sizes = np.bincount(labeled_array.ravel())
    
#     # Create a mask to remove small structures
#     remove_mask = structure_sizes < min_size
#     remove_mask[0] = 0  # Ensure the background is not removed

#     # Remove small structures
#     labeled_array[remove_mask[labeled_array]] = 0

#     # Relabel the structures after removal
#     labeled_array, num_features = measure.label(labeled_array > 0, return_num=True, connectivity=3)

#     print(f"Number of connected foreground structures after filtering: {num_features}")

#     return labeled_array

def label_foreground_structures_napari(input_array, min_size=10):
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
        18: [183, 156, 220, 255],
        19: [183, 214, 211, 255],
        20: [152, 189, 207, 255],
        22: [178, 212, 242, 255],
        23: [68, 172, 100, 255],
        24: [111, 197, 131, 255],
        25: [85, 188, 255, 255],
        26: [0, 145, 30, 255],
        27: [214, 230, 130, 255],
        29: [218, 255, 255, 255],
        30: [170, 250, 250, 255],
        31: [140, 224, 228, 255],
        32: [188, 65, 28, 255],
        33: [216, 191, 216, 255],
        34: [145, 60, 66, 255],
        35: [150, 98, 83, 255],
        39: [200, 200, 215, 255],
        40: [68, 131, 98, 255],
        42: [83, 146, 164, 255],
        44: [162, 115, 105, 255],
        46: [141, 93, 137, 255],
        48: [182, 166, 110, 255],
        50: [188, 135, 166, 255],
        52: [154, 150, 201, 255],
        54: [177, 140, 190, 255],
        56: [30, 111, 85, 255],
        58: [210, 157, 166, 255],
        60: [48, 129, 126, 255],
        62: [98, 153, 112, 255],
        64: [69, 110, 53, 255],
        65: [166, 113, 137, 255],
        66: [122, 101, 38, 255],
        68: [253, 135, 192, 255],
        69: [145, 92, 109, 255],
        70: [46, 101, 131, 255],
        73: [250, 250, 225, 255],
        74: [127, 150, 88, 255],
        76: [159, 116, 163, 255],
        78: [125, 102, 154, 255],
        80: [106, 174, 155, 255],
        82: [154, 146, 83, 255],
        84: [126, 126, 55, 255],
        85: [201, 160, 133, 255],
        87: [78, 152, 141, 255],
        89: [174, 140, 103, 255],
        91: [139, 126, 177, 255],
        93: [148, 120, 72, 255],
        95: [186, 135, 135, 255],
        97: [99, 106, 24, 255],
        98: [156, 171, 108, 255],
        100: [64, 123, 147, 255],
        102: [138, 95, 74, 255],
        103: [97, 113, 158, 255],
        104: [126, 161, 197, 255],
        105: [194, 195, 164, 255],
        107: [88, 106, 215, 255],
        115: [244, 214, 49, 255],
        116: [200, 200, 215, 255],
        118: [82, 174, 128, 255],
        119: [57, 157, 110, 255],
        120: [60, 143, 83, 255],
        121: [92, 162, 109, 255],
        122: [255, 244, 209, 255],
        126: [201, 121, 77, 255],
        127: [70, 163, 117, 255],
        128: [188, 91, 95, 255],
        130: [166, 84, 94, 255],
        131: [182, 105, 107, 255],
        132: [229, 147, 118, 255],
        134: [174, 122, 90, 255],
        136: [201, 112, 73, 255],
        138: [194, 142, 0, 255],
        140: [241, 213, 144, 255],
        141: [203, 179, 77, 255],
        143: [229, 204, 109, 255],
        145: [255, 243, 152, 255],
        147: [209, 185, 85, 255],
        149: [248, 223, 131, 255],
        151: [255, 230, 138, 255],
        152: [196, 172, 68, 255],
        153: [255, 255, 167, 255],
        154: [255, 250, 160, 255],
        155: [255, 237, 145, 255],
        156: [242, 217, 123, 255],
        158: [222, 198, 101, 255],
        160: [213, 124, 109, 255],
        161: [184, 105, 108, 255],
        162: [150, 208, 243, 255],
        163: [62, 162, 114, 255],
        166: [242, 206, 142, 255],
        167: [250, 210, 139, 255],
        168: [255, 255, 207, 255],
        170: [182, 228, 255, 255],
        171: [175, 216, 244, 255],
        172: [197, 165, 145, 255],
        174: [172, 138, 115, 255],
        176: [202, 164, 140, 255],
        177: [224, 186, 162, 255],
        179: [255, 245, 217, 255],
        180: [206, 110, 84, 255],
        181: [210, 115, 89, 255],
        182: [203, 108, 81, 255],
        183: [233, 138, 112, 255],
        184: [195, 100, 73, 255],
        185: [181, 85, 57, 255],
        186: [152, 55, 13, 255],
        187: [159, 63, 27, 255],
        188: [166, 70, 38, 255],
        189: [218, 123, 97, 255],
        190: [225, 130, 104, 255],
        191: [224, 97, 76, 255],
        193: [184, 122, 154, 255],
        194: [211, 171, 143, 255],
        195: [47, 150, 103, 255],
        197: [173, 121, 88, 255],
        198: [188, 95, 76, 255],
        199: [255, 239, 172, 255],
        200: [226, 202, 134, 255],
        201: [253, 232, 158, 255],
        202: [244, 217, 154, 255],
        203: [205, 179, 108, 255],
        205: [186, 124, 161, 255],
        207: [255, 255, 220, 255],
        208: [234, 234, 194, 255],
        209: [204, 142, 178, 255],
        210: [180, 119, 153, 255],
        211: [216, 132, 105, 255],
        212: [255, 253, 229, 255],
        213: [205, 167, 142, 255],
        214: [204, 168, 143, 255],
        215: [255, 224, 199, 255],
        217: [0, 145, 30, 255],
        218: [139, 150, 98, 255],
        219: [249, 180, 111, 255],
        220: [157, 108, 162, 255],
        221: [203, 136, 116, 255],
        222: [185, 102, 83, 255],
        224: [247, 182, 164, 255],
        226: [222, 154, 132, 255],
        227: [124, 186, 223, 255],
        228: [249, 186, 150, 255],
        230: [244, 170, 147, 255],
        231: [255, 181, 158, 255],
        232: [255, 190, 165, 255],
        233: [227, 153, 130, 255],
        234: [213, 141, 113, 255],
        236: [193, 123, 103, 255],
        237: [216, 146, 127, 255],
        238: [230, 158, 140, 255],
        239: [245, 172, 147, 255],
        241: [241, 172, 151, 255],
        243: [177, 124, 92, 255],
        244: [171, 85, 68, 255],
        245: [217, 198, 131, 255],
        246: [212, 188, 102, 255],
        247: [185, 135, 134, 255],
        249: [198, 175, 125, 255],
        250: [194, 98, 79, 255],
        250: [194, 98, 79, 255],
        251: [255, 226, 77, 255],
        252: [224, 194, 0, 255],
        253: [0, 147, 202, 255],
        254: [240, 255, 30, 255],
        255: [185, 232, 61, 255],
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