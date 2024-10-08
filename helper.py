import numpy as np
import scipy.ndimage
from qtpy.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QInputDialog, QWidget
from napari.utils.colormaps import DirectLabelColormap
from collections import defaultdict
from vispy.scene.cameras.perspective import PerspectiveCamera
from vispy.util import keys
from collections import deque
import numba
from sklearn.decomposition import PCA
import yaml
from pathlib import Path
from scipy import ndimage

from scipy.spatial import cKDTree
import os
import distinctipy

def filter_and_reassign_labels(label_data, cc_min_size):
    """
    Filter out small disconnected components within each label and reassign
    remaining labels starting from 1 and incrementing by 1.

    Parameters:
    label_data (numpy.ndarray): Input label data
    cc_min_size (int): Minimum size for a connected component to be kept

    Returns:
    numpy.ndarray: Filtered and reassigned label data
    """
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    new_label_data = np.zeros_like(label_data)
    new_label = 1

    for label in unique_labels:
        label_mask = label_data == label
        labeled_components, _ = ndimage.label(label_mask)
        
        valid_component_mask = np.zeros_like(label_mask, dtype=bool)
        
        for component in range(1, labeled_components.max() + 1):
            component_mask = labeled_components == component
            if np.sum(component_mask) >= cc_min_size:
                valid_component_mask |= component_mask

        if np.any(valid_component_mask):
            new_label_data[valid_component_mask] = new_label
            new_label += 1

    return new_label_data

@numba.jit(nopython=True)
def assign_values(papyrus_non_zero, seg_non_zero, indices, seg_mesh_data, updated_papyrus):
    for i, papyrus_coord in enumerate(papyrus_non_zero):
        seg_mesh_coord = seg_non_zero[indices[i]]
        updated_papyrus[papyrus_coord[0], papyrus_coord[1], papyrus_coord[2]] = seg_mesh_data[seg_mesh_coord[0], seg_mesh_coord[1], seg_mesh_coord[2]]
    return updated_papyrus

def assign_nearest_segmentation_values(papyrus_data, seg_mesh_data):
    """
    Optimized function to assign values to non-zero voxels in papyrus_data based on the nearest non-zero voxels in seg_mesh_data.
    
    Parameters:
    papyrus_data (np.array): 3D array of the papyrus label (chunk_size^3)
    seg_mesh_data (np.array): 3D array of the segmentation mesh ((chunk_size + pad_amount*2)^3)
    
    Returns:
    np.array: Updated papyrus data array
    """
    pad_amount = (seg_mesh_data.shape[0] - papyrus_data.shape[0]) // 2
    
    # Use boolean indexing instead of argwhere
    papyrus_mask = papyrus_data != 0
    seg_mask = seg_mesh_data != 0

    # Use np.column_stack for faster creation of coordinate arrays
    papyrus_non_zero = np.column_stack(np.nonzero(papyrus_mask))
    seg_non_zero = np.column_stack(np.nonzero(seg_mask))

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(seg_non_zero)

    # Adjust papyrus coordinates to match seg_mesh coordinates
    papyrus_non_zero_adjusted = papyrus_non_zero + pad_amount

    # Find nearest neighbors
    _, indices = tree.query(papyrus_non_zero_adjusted, workers=-1)  # Use all available cores

    # Create a new array for the updated papyrus label
    updated_papyrus = np.zeros_like(papyrus_data)

    # Use numba-optimized function for value assignment
    updated_papyrus = assign_values(papyrus_non_zero, seg_non_zero, indices, seg_mesh_data, updated_papyrus)

    return updated_papyrus

def pad_array(array, chunk_size, pad_amount=1):
    """
    Pad a 3D array with a specified number of layers of zeros if its size matches the chunk size.
    
    Args:
    array (numpy.ndarray): The input 3D array.
    chunk_size (int): The size of each dimension before padding.
    pad_amount (int): The number of layers to pad on each side. Default is 1.
    
    Returns:
    numpy.ndarray: The padded array if conditions are met, otherwise the original array.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        print("Input must be a 3D numpy array")
        return array
    
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        print("Chunk size must be a positive integer")
        return array

    if not isinstance(pad_amount, int) or pad_amount < 0:
        print("Pad amount must be a non-negative integer")
        return array
    
    if all(dim == chunk_size for dim in array.shape):
        return np.pad(array, 
                      ((pad_amount, pad_amount), 
                       (pad_amount, pad_amount), 
                       (pad_amount, pad_amount)), 
                      mode='constant', 
                      constant_values=0)
    else:
        return array

def unpad_array(array, chunk_size, pad_amount=1):
    """
    Remove the padding from a 3D array if its size is chunk_size + (2 * pad_amount) in each dimension.
    
    Args:
    array (numpy.ndarray): The input 3D array.
    chunk_size (int): The size of each dimension after unpadding.
    pad_amount (int): The number of layers padded on each side. Default is 1.
    
    Returns:
    numpy.ndarray: The unpadded array if conditions are met, otherwise the original array.
    """
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        print("Input must be a 3D numpy array")
        return array
    
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        print("Chunk size must be a positive integer")
        return array

    if not isinstance(pad_amount, int) or pad_amount < 0:
        print("Pad amount must be a non-negative integer")
        return array
    
    if all(dim == chunk_size + (2 * pad_amount) for dim in array.shape):
        return array[pad_amount:-pad_amount, 
                     pad_amount:-pad_amount, 
                     pad_amount:-pad_amount]
    else:
        return array

def find_nearest_valid_coord(num, origin=2000, chunk_size=256):
    # Convert inputs to numpy arrays for easier handling
    num = np.array(num)
    origin = np.array(origin)
    
    # Ensure num and origin have the same shape
    if num.shape != origin.shape:
        if origin.size == 1:
            origin = np.full_like(num, origin)
        elif num.size == 1:
            num = np.full_like(origin, num)
        else:
            raise ValueError("num and origin must have the same shape or one must be a scalar")

    difference = num - origin
    quotient = difference // chunk_size
    remainder = difference % chunk_size

    result = np.where(remainder == 0, num,
                      np.where(remainder > chunk_size / 2, 
                               origin + chunk_size * (quotient + 1),
                               origin + chunk_size * quotient))
    
    lower_bound = origin - (7 * chunk_size)
    return np.maximum(result, lower_bound)

def is_valid_coord(num_or_list, origin=2000, chunk_size=256):
    # Ensure origin is a list or numpy array
    if isinstance(origin, (int, float)):
        origin = [origin] * (3 if isinstance(num_or_list, (list, np.ndarray)) else 1)
    origin = np.array(origin)

    if isinstance(num_or_list, (int, float)):
        difference = abs(num_or_list - origin[0])
        return difference % chunk_size == 0
    elif isinstance(num_or_list, (list, np.ndarray)):
        num_or_list = np.array(num_or_list)
        if len(num_or_list) != len(origin):
            raise ValueError("Length of input and origin must match")
        differences = np.abs(num_or_list - origin)
        return (differences % chunk_size == 0).all()
    else:
        raise TypeError("Input must be a number, list, or numpy array")

def read_config(config_path='napari_config.yaml'):
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config.get('cube_info',{}), config.get('customizable_hotkeys', {})
    return {}

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
            #     fov = self._event_value[1] - d[1] / 50.0
            #     self.fov = min(180.0, max(0.0, fov))

    except AttributeError as e:
        print(f"AttributeError in patched_viewbox_mouse_event: {e}")
    except ValueError as e:
        print(f"ValueError in patched_viewbox_mouse_event: {e}")
    except TypeError as e:
        print(f"TypeError in patched_viewbox_mouse_event: {e}")
    except Exception as e:
        print(f"Unexpected error in patched_viewbox_mouse_event: {e}")

def find_center_voxel(mask):
    # Ensure the mask is boolean
    mask = mask.astype(bool)

    # Calculate the center of mass
    center_of_mass = ndimage.center_of_mass(mask)

    # Round to nearest integer to get voxel coordinates
    center_voxel = np.round(center_of_mass).astype(int)

    # Return as z, y, x
    return tuple(center_voxel)

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
    foreground = input_array > 0

    # Label connected components
    labeled_array, num_features = scipy.ndimage.label(foreground)
    print(f"Number of connected foreground structures before filtering: {num_features}")
    
    # Measure the size of each connected component
    structure_sizes = np.array(scipy.ndimage.sum(foreground, labeled_array, range(num_features + 1)))
    small_structure_sizes = structure_sizes[structure_sizes < 10]

    # Create a mask to remove small structures
    remove_mask = structure_sizes < min_size
    remove_mask[0] = 0  # Ensure the background is not removed

    # Remove small structures
    labeled_array[remove_mask[labeled_array]] = 0

    # Relabel the structures after removal
    labeled_array, num_features = scipy.ndimage.label(labeled_array > 0)

    print(f"Number of connected foreground structures after filtering: {np.max(labeled_array)}")

    return labeled_array

def generate_light_colormap(N):
    base_colormap = get_light_slicer_colormap()
    colors = list(base_colormap.values())
    if N > 30:
        generated_colors = distinctipy.get_colors(N - 30, pastel_factor=0.7)
        colors_255 = [[int(c * 255) for c in generated_color] + [255] for generated_color in generated_colors]
        colors.extend(colors_255)
    # colors = distinctipy.get_colors(N, pastel_factor=0.7)
    print(colors)
    # colors_255 = [[int(c * 255) for c in color] + [255] for color in colors]
    # print(colors_255)
    return {i: color for i, color in enumerate(colors)}

def get_light_slicer_colormap():
    return {
        0: [0, 0, 0, 0],
        1: [128, 174, 128, 255],
        2: [241, 214, 145, 255],
        3: [177, 122, 101, 255],
        4: [111, 184, 210, 255],
        5: [216, 101, 79, 255],
        6: [221, 170, 101, 255],
        7: [144, 238, 144, 255],
        8: [255, 181, 158, 255],
        9: [220, 245, 20, 255],
        10: [78, 63, 0, 255],
        11: [255, 250, 220, 255],
        12: [230, 220, 70, 255],
        13: [200, 200, 235, 255],
        14: [82, 174, 128, 255],
        15: [244, 214, 49, 255],
        16: [0, 151, 206, 255],
        17: [185, 232, 61, 255],
        18: [183, 156, 220, 255],
        19: [183, 214, 211, 255],
        20: [152, 189, 207, 255],
        21: [10, 255, 170, 255],
        22: [178, 212, 242, 255],
        23: [68, 172, 100, 255],
        24: [111, 197, 131, 255],
        25: [85, 188, 255, 255],
        26: [0, 145, 30, 255],
        27: [214, 230, 130, 255],
        28: [0, 147, 202, 255],
        29: [218, 255, 255, 255],
        30: [170, 250, 250, 255]
    }

def get_slicer_colormap():
    return {
        0: [0, 0, 0, 0],
        1: [128, 174, 128, 255],
        2: [241, 214, 145, 255],
        3: [177, 122, 101, 255],
        4: [111, 184, 210, 255],
        5: [216, 101, 79, 255],
        6: [221, 170, 101, 255],
        7: [144, 238, 144, 255],
        8: [255, 181, 158, 255],
        9: [220, 245, 20, 255],
        10: [78, 63, 0, 255],
        11: [255, 250, 220, 255],
        12: [230, 220, 70, 255],
        13: [200, 200, 235, 255],
        14: [82, 174, 128, 255],
        15: [244, 214, 49, 255],
        16: [0, 151, 206, 255],
        17: [185, 232, 61, 255],
        18: [183, 156, 220, 255],
        19: [183, 214, 211, 255],
        20: [152, 189, 207, 255],
        21: [10, 255, 170, 255],
        22: [178, 212, 242, 255],
        23: [68, 172, 100, 255],
        24: [111, 197, 131, 255],
        25: [85, 188, 255, 255],
        26: [0, 145, 30, 255],
        27: [214, 230, 130, 255],
        28: [0, 147, 202, 255],
        29: [218, 255, 255, 255],
        30: [170, 250, 250, 255],
        31: [140, 224, 228, 255],
        32: [188, 65, 28, 255],
        33: [216, 191, 216, 255],
        34: [145, 60, 66, 255],
        35: [150, 98, 83, 255],
        36: [200, 200, 215, 255],
        37: [68, 131, 98, 255],
        38: [83, 146, 164, 255],
        39: [162, 115, 105, 255],
        40: [141, 93, 137, 255],
        41: [182, 166, 110, 255],
        42: [188, 135, 166, 255],
        43: [154, 150, 201, 255],
        44: [177, 140, 190, 255],
        45: [30, 111, 85, 255],
        46: [210, 157, 166, 255],
        47: [48, 129, 126, 255],
        48: [98, 153, 112, 255],
        49: [69, 110, 53, 255],
        50: [166, 113, 137, 255],
        51: [122, 101, 38, 255],
        52: [253, 135, 192, 255],
        53: [145, 92, 109, 255],
        54: [46, 101, 131, 255],
        55: [250, 250, 225, 255],
        56: [127, 150, 88, 255],
        57: [159, 116, 163, 255],
        58: [125, 102, 154, 255],
        59: [106, 174, 155, 255],
        60: [154, 146, 83, 255],
        61: [126, 126, 55, 255],
        62: [201, 160, 133, 255],
        63: [78, 152, 141, 255],
        64: [174, 140, 103, 255],
        65: [139, 126, 177, 255],
        66: [148, 120, 72, 255],
        67: [186, 135, 135, 255],
        68: [99, 106, 24, 255],
        69: [156, 171, 108, 255],
        70: [64, 123, 147, 255],
        71: [138, 95, 74, 255],
        72: [97, 113, 158, 255],
        73: [126, 161, 197, 255],
        74: [194, 195, 164, 255],
        75: [88, 106, 215, 255],
        76: [244, 214, 49, 255],
        77: [200, 200, 215, 255],
        78: [241, 172, 151, 255],
        79: [57, 157, 110, 255],
        80: [60, 143, 83, 255],
        81: [92, 162, 109, 255],
        82: [255, 244, 209, 255],
        83: [201, 121, 77, 255],
        84: [70, 163, 117, 255],
        85: [188, 91, 95, 255],
        86: [166, 84, 94, 255],
        87: [182, 105, 107, 255],
        88: [229, 147, 118, 255],
        89: [174, 122, 90, 255],
        90: [201, 112, 73, 255],
        91: [194, 142, 0, 255],
        92: [241, 213, 144, 255],
        93: [203, 179, 77, 255],
        94: [229, 204, 109, 255],
        95: [255, 243, 152, 255],
        96: [209, 185, 85, 255],
        97: [248, 223, 131, 255],
        98: [255, 230, 138, 255],
        99: [196, 172, 68, 255],
        100: [255, 255, 167, 255],
        101: [255, 250, 160, 255],
        102: [255, 237, 145, 255],
        103: [242, 217, 123, 255],
        104: [222, 198, 101, 255],
        105: [213, 124, 109, 255],
        106: [184, 105, 108, 255],
        107: [150, 208, 243, 255],
        108: [62, 162, 114, 255],
        109: [242, 206, 142, 255],
        110: [250, 210, 139, 255],
        111: [255, 255, 207, 255],
        112: [182, 228, 255, 255],
        113: [175, 216, 244, 255],
        114: [197, 165, 145, 255],
        115: [172, 138, 115, 255],
        116: [202, 164, 140, 255],
        117: [224, 186, 162, 255],
        118: [255, 245, 217, 255],
        119: [206, 110, 84, 255],
        120: [210, 115, 89, 255],
        121: [203, 108, 81, 255],
        122: [233, 138, 112, 255],
        123: [195, 100, 73, 255],
        124: [181, 85, 57, 255],
        125: [152, 55, 13, 255],
        126: [159, 63, 27, 255],
        127: [166, 70, 38, 255],
        128: [218, 123, 97, 255],
        129: [225, 130, 104, 255],
        130: [224, 97, 76, 255],
        131: [184, 122, 154, 255],
        132: [211, 171, 143, 255],
        133: [47, 150, 103, 255],
        134: [173, 121, 88, 255],
        135: [188, 95, 76, 255],
        136: [255, 239, 172, 255],
        137: [226, 202, 134, 255],
        138: [253, 232, 158, 255],
        139: [244, 217, 154, 255],
        140: [205, 179, 108, 255],
        141: [186, 124, 161, 255],
        142: [255, 255, 220, 255],
        143: [234, 234, 194, 255],
        144: [204, 142, 178, 255],
        145: [180, 119, 153, 255],
        146: [216, 132, 105, 255],
        147: [255, 253, 229, 255],
        148: [205, 167, 142, 255],
        149: [204, 168, 143, 255],
        150: [255, 224, 199, 255],
        151: [0, 145, 30, 255],
        152: [139, 150, 98, 255],
        153: [249, 180, 111, 255],
        154: [157, 108, 162, 255],
        155: [203, 136, 116, 255],
        156: [185, 102, 83, 255],
        157: [247, 182, 164, 255],
        158: [222, 154, 132, 255],
        159: [124, 186, 223, 255],
        160: [249, 186, 150, 255],
        161: [244, 170, 147, 255],
        162: [192, 104, 88, 255],
        163: [255, 190, 165, 255],
        164: [227, 153, 130, 255],
        165: [213, 141, 113, 255],
        166: [193, 123, 103, 255],
        167: [216, 146, 127, 255],
        168: [230, 158, 140, 255],
        169: [245, 172, 147, 255],
        170: [250, 250, 210, 255],
        171: [177, 124, 92, 255],
        172: [171, 85, 68, 255],
        173: [217, 198, 131, 255],
        174: [212, 188, 102, 255],
        175: [185, 135, 134, 255],
        176: [198, 175, 125, 255],
        177: [194, 98, 79, 255],
        178: [194, 98, 79, 255],
        179: [255, 226, 77, 255],
        180: [224, 194, 0, 255],
        181: [0, 147, 202, 255],
        182: [240, 255, 30, 255],
        183: [185, 232, 61, 255],
    }

def load_or_create_colormap():
    current_directory = os.getcwd()
    colormap_file = os.path.join(current_directory, 'colormap.yaml')
    
    if os.path.exists(colormap_file):
        with open(colormap_file, 'r') as file:
            colormap = yaml.safe_load(file)
    else:
        colormap = generate_light_colormap(200)
        with open(colormap_file, 'w') as file:
            yaml.dump(colormap, file)
    
    return colormap

from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtGui import QColor

def edit_label_color(active_label):
    colormap = load_or_create_colormap()
    
    current_color = colormap.get(active_label, [255, 255, 255, 255])
    initial_color = QColor(*current_color)
    
    color_dialog = QColorDialog()
    color_dialog.setCurrentColor(initial_color)
    
    if color_dialog.exec_():
        new_color = color_dialog.currentColor()
        new_color_rgba = [new_color.red(), new_color.green(), new_color.blue(), new_color.alpha()]
        
        colormap[active_label] = new_color_rgba
        
        current_directory = os.getcwd()
        colormap_file = os.path.join(current_directory, 'colormap.yaml')
        
        with open(colormap_file, 'w') as file:
            yaml.dump(colormap, file)
        
        print(f"Updated color for label {active_label}: {new_color_rgba}")
    else:
        print("Color selection cancelled")

def reset_colormap():
    original_colormap = get_slicer_colormap()
    current_directory = os.getcwd()
    colormap_file = os.path.join(current_directory, 'colormap.yaml')
    
    with open(colormap_file, 'w') as file:
        yaml.dump(original_colormap, file)
    
    print("Colormap reset to original Slicer colormap")
    return get_direct_label_colormap()

def get_direct_label_colormap():
    colormap = load_or_create_colormap()
    
    # Normalize colors to 0-1 range
    normalized_colormap = defaultdict(lambda: np.array([0, 0, 0, 0]))
    for k, color in colormap.items():
        normalized_colormap[k] = np.array(color) / 255
    
    # Add None key with a default color (e.g., transparent black)
    normalized_colormap[None] = np.array([0, 0, 0, 0])
    
    # Create the DirectLabelColormap
    return DirectLabelColormap(color_dict=normalized_colormap)