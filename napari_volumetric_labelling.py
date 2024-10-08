import napari
import os
import numpy as np
from helper import *
from gui_components import VesuviusGUI
from napari.layers import Image
from PyQt5.QtCore import QTimer
from scipy.ndimage import binary_erosion
from scipy import ndimage
from qtpy.QtWidgets import QInputDialog, QMessageBox
from qtpy.QtCore import QTimer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
import sys
from collections import namedtuple
from vispy.scene.cameras.perspective import Base3DRotationCamera
from vispy.util import keys
import asyncio
from data_manager import DataManager
from config import Config  # Assuming you create a Config class
import warnings
import magicgui
# import fastmorph


Base3DRotationCamera.viewbox_mouse_event = patched_viewbox_mouse_event
warnings.filterwarnings("ignore", message="Contours are not displayed during 3D rendering")
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
warnings.filterwarnings("ignore", message="Refusing to run a QApplication with no topLevelWidgets.")

config_path = 'local_napari_config.yaml' if os.path.exists('local_napari_config.yaml') else 'napari_config.yaml'
# config_path = 'ink_det_config.yaml' if os.path.exists('ink_det_config.yaml') else 'napari_config.yaml'

config = Config(config_path)
# if __name__ == "__main__":
scroll_name = config.cube_config.scroll_name
zyx = config.cube_config.zyx
z = config.cube_config.z
y = config.cube_config.y
x = config.cube_config.x
z_num = int(z)
y_num = int(y)
x_num = int(x)

# Check if the initial coordinates are valid and align them if not
origin_val = 0
if scroll_name == 's1' or scroll_name == 's1_8um':
    origin_val = 2000
if config.cube_config.align_coordinates and not is_valid_coord([z_num, y_num, x_num], origin_val, config.cube_config.chunk_size):
    print(f"Invalid coordinates: {z_num}, {y_num}, {x_num}")
    z_num = find_nearest_valid_coord(z_num, origin_val, config.cube_config.chunk_size)
    y_num = find_nearest_valid_coord(y_num, origin_val, config.cube_config.chunk_size)
    x_num = find_nearest_valid_coord(x_num, origin_val, config.cube_config.chunk_size)
    print(f"Using nearest valid coordinates: {z_num}, {y_num}, {x_num}")
    z = str(z_num).zfill(5)
    y = str(y_num).zfill(5)
    x = str(x_num).zfill(5)
    config.cube_config.update_coordinates(z=z, y=y, x=x)

print('Creating DataManager')
data_manager = DataManager(config.cube_config)

# Data location and size parameters
papyrus_mask_factor = config.cube_config.factor

chunk_size = config.cube_config.chunk_size
global pad_amount
pad_amount = config.cube_config.pad_amount
brush_size = config.cube_config.brush_size
author = config.cube_config.author
main_label_layer_name = config.cube_config.main_label_layer_name

current_directory = os.getcwd()

pad_state = False

data = data_manager.raw_data
label_data = data_manager.label_data
original_data = data_manager.original_label_data
ink_pred_data = data_manager.original_ink_pred_data
voxelized_segmentation_mesh_data = data_manager.voxelized_segmentation_mesh_data

# Initialize the Napari viewer
print('Initializing Napari Viewer')
viewer = napari.Viewer()

#layer name variables
papyrus_label_name = 'Papyrus Labels'
ink_label_name = 'Ink Labels'
data_name = 'Data'
seg_mesh_name = 'Segmentation Mesh'
ff_name = 'Flood Fill'
cc_preview_name = 'Connected Components Preview'
label_3d_name = '3D Label Edit Layer'
pad_state = False
cut_side = True
plane_shift_status = False
eraser_size = 4

global erase_mode
erase_mode = False
global erase_slice_width
erase_slice_width = 30

if (main_label_layer_name != 'ink' and label_data is not None) or ink_pred_data is None:
    main_label_name = papyrus_label_name
else: 
    main_label_name = ink_label_name

# Add the 3D data to the viewer
image_layer =  viewer.add_image(data, colormap='gray', name=data_name)
if label_data is not None:
    papyrus_label_layer = viewer.add_labels(label_data, name=papyrus_label_name)
if original_data is not None:
    test_layer = viewer.add_labels(original_data, name="Original Semantic Mask")
if ink_pred_data is not None:
    ink_labels_layer = viewer.add_labels(ink_pred_data, name=ink_label_name)
if voxelized_segmentation_mesh_data is not None:
    seg_mesh_layer = viewer.add_labels(voxelized_segmentation_mesh_data, name=seg_mesh_name)
    offset = np.array([config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount])
    seg_mesh_layer.translate = -offset
    seg_mesh_layer.opacity = 1

@magicgui.magicgui(auto_call=False, on={"visible": False})
def toggle_smooth_labels(viewer: napari.viewer.Viewer, layer: napari.layers.Labels, on=False):
    if viewer.dims.ndisplay != 3:
        return
    try:
        print("Toggling smooth labels for layer:", layer.name)
        node = viewer.window.qt_viewer.layer_to_visual[layer].node
        node.iso_gradient = not node.iso_gradient
        if on:
            node.iso_gradient = True
    except Exception as e:
        print("Error toggling smooth labels:", e)
    

# Functions:
def isolate_selected_label(viewer):
    global previous_label_3d_data, manual_changes_mask, pad_state, pad_amount

    if label_3d_name in viewer.layers:
        # If 3D edit layer exists, apply changes and remove it
        update_label_from_3d_edit_layer(viewer)
        viewer.layers.remove(label_3d_name)
        viewer.layers[main_label_name].visible = True
        viewer.layers[main_label_name].refresh()
        print(f"Changes applied to {main_label_name} layer and 3D edit layer removed.")
        viewer.layers.selection.active = viewer.layers[main_label_name]
    else:
        if viewer.layers.selection.active is not None:
            active_mode = viewer.layers.selection.active.mode
        else:
            active_mode = 'pan_zoom'

        # Update main_label_name layer from label_3d_name layer if it exists
        update_label_from_3d_edit_layer(viewer)

        # Get the selected label
        selected_label = viewer.layers[main_label_name].selected_label
        if selected_label == 0:
            print("No label selected. Please select a label first.")
            return

        # Create a copy of the label data and isolate the selected label
        new_label_data = viewer.layers[main_label_name].data.copy()
        new_label_data[new_label_data != selected_label] = 0

        # Setup the label_3d_name layer
        new_label_layer = setup_label_3d_layer(viewer, new_label_data, active_mode)
        new_label_layer.selected_label = selected_label

        # Store the current state of the label_3d_name layer for future comparison
        previous_label_3d_data = new_label_data.copy()

        viewer.layers[main_label_name].visible = False
        new_label_layer.visible = True

def toggle_preserve_labels(viewer):
    active_layer = viewer.layers.selection.active
    if isinstance(active_layer, napari.layers.Labels):
        active_layer.preserve_labels = not active_layer.preserve_labels
        status = "enabled" if active_layer.preserve_labels else "disabled"
 
def toggle_show_selected_label(viewer):
    active_layer = viewer.layers.selection.active
    if isinstance(active_layer, napari.layers.Labels):
        active_layer.show_selected_label = not active_layer.show_selected_label
        status = "enabled" if active_layer.show_selected_label else "disabled"

def move_seg_mesh_label(viewer, dz=1, dy=0, dx=0, move_all=False):
    if seg_mesh_name not in viewer.layers:
        print(f"Error: {seg_mesh_name} layer not found in the viewer.")
        return

    seg_mesh_layer = viewer.layers[seg_mesh_name]
    seg_mesh_data = seg_mesh_layer.data.copy()

    if move_all:
        # Move all non-zero labels
        non_zero_mask = seg_mesh_data != 0
        new_seg_mesh_data = np.zeros_like(seg_mesh_data)
        
        z, y, x = np.nonzero(non_zero_mask)
        new_z = z + dz
        new_y = y + dy
        new_x = x + dx
        
        within_bounds = (
            (new_z >= 0) & (new_z < seg_mesh_data.shape[0]) &
            (new_y >= 0) & (new_y < seg_mesh_data.shape[1]) &
            (new_x >= 0) & (new_x < seg_mesh_data.shape[2])
        )
        
        new_seg_mesh_data[new_z[within_bounds], new_y[within_bounds], new_x[within_bounds]] = seg_mesh_data[z[within_bounds], y[within_bounds], x[within_bounds]]
        seg_mesh_layer.data = new_seg_mesh_data
        print(f"All labels moved by (dz={dz}, dy={dy}, dx={dx}).")
    else:
        selected_label = seg_mesh_layer.selected_label

        if selected_label == 0:
            print("Error: No label selected. Please select a label to move.")
            return

        label_mask = seg_mesh_data == selected_label
        seg_mesh_data[label_mask] = 0
        z, y, x = np.nonzero(label_mask)

        new_z = z + dz
        new_y = y + dy
        new_x = x + dx
        within_bounds = (
            (new_z >= 0) & (new_z < seg_mesh_data.shape[0]) &
            (new_y >= 0) & (new_y < seg_mesh_data.shape[1]) &
            (new_x >= 0) & (new_x < seg_mesh_data.shape[2])
        )

        seg_mesh_data[new_z[within_bounds], new_y[within_bounds], new_x[within_bounds]] = selected_label
        seg_mesh_layer.data = seg_mesh_data

        print(f"Label {selected_label} moved by (dz={dz}, dy={dy}, dx={dx}).")


def color_semantic_mask(viewer):
    print("Coloring semantic mask.")
    colored_papyrus_label = assign_nearest_segmentation_values(viewer.layers[papyrus_label_name].data, viewer.layers[seg_mesh_name].data)
    viewer.layers[papyrus_label_name].data = colored_papyrus_label

def align_cube_with_selected_label(viewer, new_chunk_size=config.cube_config.edit_chunk_size):
    if viewer.layers[main_label_name].visible:
        align_layer = viewer.layers[main_label_name]
    elif ink_pred_data is None or (not viewer.layers[ink_label_name].visible and viewer.layers[papyrus_label_name].visible):
        align_layer = viewer.layers[papyrus_label_name]
    else:
        align_layer = viewer.layers[ink_label_name]
    
    # Get the selected label value
    selected_label = align_layer.selected_label
    if selected_label == 0:
        print("No label selected.")
        return
    
    masked_labels = (align_layer.data == selected_label).astype(np.uint8)
    center_voxel = find_center_voxel(masked_labels)

    center_voxel = np.array(center_voxel)
    viewer.layers[data_name].plane.position = (new_chunk_size // 2, new_chunk_size // 2, new_chunk_size // 2)
    center_voxel[0] += config.cube_config.z_num - (new_chunk_size // 2)
    center_voxel[1] += config.cube_config.y_num - (new_chunk_size // 2)
    center_voxel[2] += config.cube_config.x_num - (new_chunk_size // 2)

    update_and_reload_data(viewer, data_manager, config, center_voxel[0], center_voxel[1], center_voxel[2], new_chunk_size)

def align_plane_with_selected_label(viewer):
    if viewer.layers[main_label_name].visible:
        align_layer = viewer.layers[main_label_name]
    elif ink_pred_data is None or (not viewer.layers[ink_label_name].visible and viewer.layers[papyrus_label_name].visible):
        align_layer = viewer.layers[papyrus_label_name]
    else:
        align_layer = viewer.layers[ink_label_name]
    data_layer = viewer.layers[data_name]
    
    # Check if data layer is visible and depicted as a plane
    if not data_layer.visible or data_layer.depiction != 'plane':
        print("Data layer is not visible or not depicted as a plane.", data_layer.visible, data_layer.depiction)
        return
    
    # Get the selected label value
    selected_label = align_layer.selected_label
    if selected_label == 0:
        print("No label selected.")
        return
    
    masked_labels = (align_layer.data == selected_label).astype(np.uint8)
    position, normal = find_best_intersecting_plane_napari(masked_labels)
    data_layer.plane.position = position
    data_layer.plane.normal = normal

@viewer.mouse_drag_callbacks.append
def pan_with_middle_mouse(viewer, event):
    if event.button == 3 or event.button == 2 or (event.button == 1 and keys.SHIFT in event.modifiers):  # Middle mouse button or right click
        if viewer.layers.selection.active is None:
            viewer.layers.selection.active = viewer.layers[main_label_name]
            viewer.layers[main_label_name].mode = 'pan_zoom'
        original_mode = viewer.layers.selection.active.mode
        viewer.layers.selection.active.mode = 'pan_zoom'
        yield
        while event.type == 'mouse_move':
            yield
        viewer.layers.selection.active.mode = original_mode

def apply_global_brush_size(viewer, source_layer=None):
    global brush_size
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels) and layer != source_layer:
            if layer.brush_size != brush_size:
                layer.brush_size = brush_size

def update_global_brush_size(event):
    global brush_size
    # The new brush size is stored in the layer, not in the event
    brush_size = event.source.brush_size
    # Pass the source layer to avoid updating it again
    apply_global_brush_size(viewer, source_layer=event.source)

def setup_brush_size_listener(viewer, layer_name):
    brush_layer = viewer.layers[layer_name]
    brush_layer.events.brush_size.connect(update_global_brush_size)

def switch_to_data_layer(viewer):
    viewer.layers[data_name].visible = True
    viewer.layers.selection.active = viewer.layers[data_name]

def toggle_labels_visibility(viewer):
    msg = 'toggle labels visibility'
    viewer.status = msg
    print(msg)
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers:
        viewer.layers[label_3d_name].visible = not viewer.layers[label_3d_name].visible
    else:
        viewer.layers[main_label_name].visible = not viewer.layers[main_label_name].visible

def toggle_data_visibility(viewer):
    msg = 'toggle data visibility'
    viewer.status = msg
    print(msg)
    image_layer.visible = not image_layer.visible

def decrease_brush_size(viewer):
    global brush_size
    msg = 'decrease brush size'
    viewer.status = msg
    print(msg)
    brush_size -= 1
    apply_global_brush_size(viewer)

def increase_brush_size(viewer):
    global brush_size
    msg = 'increase brush size'
    viewer.status = msg
    print(msg)
    brush_size += 1
    apply_global_brush_size(viewer)

# Add an empty labels layer for the flood fill result
flood_fill_layer = None

def flood_fill(viewer, distance=20):
    if flood_fill_layer == None:
        flood_fill_layer = viewer.add_labels(np.zeros_like(data), name=ff_name)
    # Get the cursor position in data coordinates
    cursor_position = viewer.cursor.position
    cursor_position = tuple(int(np.round(coord)) for coord in cursor_position)

    # Get the current labels layer
    if papyrus_label_layer in viewer.layers:
        label_layer = viewer.layers[papyrus_label_name]
    elif ink_labels_layer in viewer.layers:
        label_layer = viewer.layers[ink_label_name]

    # Get the current labels
    labels = label_layer.data

    # Perform the flood fill operation
    flood_fill_result = limited_bfs_flood_fill(labels, cursor_position, distance)

    # Update the flood fill layer with the result
    flood_fill_layer.data = flood_fill_result

def large_flood_fill(viewer):
    flood_fill(viewer, 100)

# Variable to store the previous oblique plane information
prev_plane_info_var = None
prev_erase_plane_info_var = None

# Define a namedtuple to store camera information
CameraInfo = namedtuple('CameraInfo', ['center', 'angles', 'zoom'])
global prev_camera_pos
prev_camera_pos = None

# Persistent variables to store the previous state and mask
previous_label_3d_data = None
manual_changes_mask = None

def get_current_plane_info(viewer):
    position = viewer.layers[data_name].plane.position
    normal = viewer.layers[data_name].plane.normal
    return {'position': position, 'normal': normal}

def get_current_camera_info(viewer):
    center=viewer.camera.center
    angles=viewer.camera.angles
    zoom=viewer.camera.zoom
    return CameraInfo(center, angles, zoom)

def set_camera_view(viewer, camera_pos = None):
    if camera_pos is not None:
        viewer.camera.center = camera_pos.center
        viewer.camera.angles = camera_pos.angles
        viewer.camera.zoom = camera_pos.zoom
    else:
        center = (chunk_size // 2, chunk_size // 2, chunk_size // 2)
        viewer.camera.center = center
        viewer.camera.angles = (-10,30,65)  # Setting roll to 0
        viewer.camera.zoom = 1.8  # You may need to adjust this value

def process_value(value, data, erode, erosion_iterations, dilation_iterations, original_label_data):
    structure_mask = data == value
    result = np.zeros_like(data, dtype=np.uint8)
    
    if erode:
        padded_structure = np.pad(structure_mask, pad_width=erosion_iterations, mode='constant', constant_values=value)
        eroded_padded_structure = binary_erosion(padded_structure, iterations=erosion_iterations)
        eroded_structure = eroded_padded_structure[
            erosion_iterations:-erosion_iterations,
            erosion_iterations:-erosion_iterations,
            erosion_iterations:-erosion_iterations
        ]
        if eroded_structure.shape != structure_mask.shape:
            eroded_structure = np.zeros_like(structure_mask)
        result[eroded_structure] = value
    else:
        if dilation_iterations > 0:
            # Create a mask of all other values
            other_values_mask = (data != 0) & (data != value)
            
            dilated_structure = numba_dilation_3d_labels(structure_mask, dilation_iterations)
            # Remove areas where dilation intersects with other values
            final_dilated_structure = dilated_structure & ~other_values_mask
            
            # Ensure dilation doesn't exceed the original label data boundaries
            if original_label_data is not None:
                final_dilated_structure = final_dilated_structure & (original_label_data != 0)
            
            # Apply the result
            result[final_dilated_structure] = value
        else:
            result[structure_mask] = value
    
    return result

def erode_dilate_selected_label(data, label_value, erode=True, iterations=1, original_mask=None, main_label_data=None):
    mask = data == label_value
    if main_label_data is None:
        other_labels_mask = (data != 0) & (data != label_value)
    else:
        other_labels_mask = (main_label_data != 0) & (main_label_data != label_value)

    if erode:
        result = binary_erosion(mask, iterations=iterations)
    else:
        temp_dilated = numba_dilation_3d_labels(mask, iterations)

        # Ensure dilation doesn't expand into other labels
        result = temp_dilated & ~other_labels_mask

        # If original_mask is provided, ensure dilation doesn't exceed it
        if original_mask is not None:
            result = result & original_mask
    new_data = data.copy()
    new_data[mask] = 0  # Remove the original label
    new_data[result] = label_value  # Apply the new label
    return new_data

@thread_worker
def erode_dilate_labels_worker(data, erode=True, erosion_iterations=1, dilation_iterations=1, original_label_data=data_manager.original_label_data):
    unique_values = np.unique(data[(data > 0) & (data < 254)])
    result = np.zeros_like(data, dtype=np.uint8)
    
    total_values = len(unique_values)
    for i, value in enumerate(unique_values):
        partial_result = process_value(value, data, erode, erosion_iterations, dilation_iterations, original_label_data)
        result = np.maximum(result, partial_result)
        yield i / total_values  # This will update the progress bar
    
    return result

def erode_dilate_labels(viewer, data, erode=True, erosion_iterations=1, dilation_iterations=1):
    original_label_data = data_manager.original_label_data
    worker = erode_dilate_labels_worker(data, erode, erosion_iterations, dilation_iterations, original_label_data=original_label_data)
    
    def update_progress(progress):
        show_info(f"Processing: {progress:.0%}")
    
    def on_complete(result):
        viewer.layers[main_label_name].data = result
        show_info("Processing complete!")
    
    worker.yielded.connect(update_progress)
    worker.returned.connect(on_complete)
    worker.start()

def erode_labels(viewer):
    selected_option = gui.label_selection_combo.currentText()
    active_layer = viewer.layers.selection.active

    if "Selected Label" in selected_option:
        if not isinstance(active_layer, napari.layers.Labels) or active_layer.selected_label == 0:
            show_popup("Please select a valid label on an active label layer.")
            return

    if "No Warning" not in selected_option:
        if "Selected Label" in selected_option:
            msg = f"Are you sure you want to erode the selected label ({active_layer.selected_label})? This operation cannot be undone."
        else:
            msg = "Are you sure you want to erode all labels? This operation cannot be undone."

        response = confirm_popup(msg)
        if response != QMessageBox.Yes:
            print('eroding labels cancelled')
            return 

    if "Selected Label" in selected_option:
        new_data = erode_dilate_selected_label(active_layer.data, active_layer.selected_label, erode=True)
        active_layer.data = new_data
    else:
        erode_dilate_labels(viewer, viewer.layers[main_label_name].data)
    
    viewer.layers[main_label_name].refresh()

def dilate_labels(viewer):
    selected_option = gui.label_selection_combo.currentText()
    active_layer = viewer.layers.selection.active

    if "Selected Label" in selected_option:
        if not isinstance(active_layer, napari.layers.Labels) or active_layer.selected_label == 0:
            show_popup("Please select a valid label on an active label layer.")
            return

    if "No Warning" not in selected_option:
        if "Selected Label" in selected_option:
            msg = f"Are you sure you want to dilate the selected label ({active_layer.selected_label})? This operation cannot be undone."
        else:
            msg = "Are you sure you want to dilate all labels? This operation cannot be undone. It will only dilate up to the borders of the original mask file."

        response = confirm_popup(msg)
        if response != QMessageBox.Yes:
            print('dilating labels cancelled')
            return 

    if "Selected Label" in selected_option:
        original_mask = data_manager.original_label_data != 0
        # main_label_data = viewer.layers[main_label_name].data
        main_label_data = active_layer.data
        new_data = erode_dilate_selected_label(active_layer.data, active_layer.selected_label, erode=False, original_mask=original_mask, main_label_data=main_label_data)
        active_layer.data = new_data
    else:
        erode_dilate_labels(viewer, viewer.layers[main_label_name].data, erode=False)
    
    viewer.layers[main_label_name].refresh()

def morphological_tunnel_fill(viewer):
    

    # Prompt user for radius input
    radius, ok = QInputDialog.getInt(
        viewer.window._qt_window,
        "Tunnel Fill",
        "Enter the radius for tunnel fill:",
        value=5,
        min=1,
        max=1000
    )

    if not ok:
        print("Tunnel fill operation cancelled.")
        return
    
    active_layer = viewer.layers.selection.active
    if not isinstance(active_layer, napari.layers.Labels):
        show_popup("Please select a label layer.")
        return
    
    label_val = active_layer.selected_label
    if label_val == 0:
        show_popup("Please select a non-zero label.")
        return

    # Confirm operation
    confirm = QMessageBox.question(
        viewer.window._qt_window,
        "Confirm Tunnel Fill",
        f"Are you sure you want to perform tunnel fill with radius {radius} on the selected label ({label_val})? This operation cannot be undone, consider saving first.",
        QMessageBox.Yes | QMessageBox.No
    )

    if confirm == QMessageBox.No:
        print("Tunnel fill operation cancelled.")
        return

    arr = active_layer.data.copy()
    arr = (arr == label_val).astype(bool)
    arr = np.pad(arr, pad_width=radius, mode='constant', constant_values=0)


    # arr = fastmorph.spherical_dilate(arr, radius=radius, parallel=4, in_place=True)
    arr = numba_dilation_3d_labels(arr, radius)
    arr = binary_erosion(arr, iterations=radius)
    # arr = fastmorph.spherical_erode(arr, radius=radius, parallel=4, in_place=True)

    arr = arr[radius:-radius, radius:-radius, radius:-radius]
    arr = np.where(arr > 0, label_val, 0)
    # Update the active layer data
    mask = (active_layer.data == 0) & (arr != 0)
    active_layer.data[mask] = arr[mask]
    active_layer.refresh()
    show_popup("Tunnel fill operation completed.")
    # return arr

def shift_prev_erase_plane(direction):
    global erase_slice_width, prev_erase_plane_info_var
    if prev_erase_plane_info_var is not None:
        current_position = np.array(prev_erase_plane_info_var['position'])
        normal_vector = np.array(prev_erase_plane_info_var['normal'])
        normal_vector /= np.linalg.norm(normal_vector)
        new_position = current_position + direction * normal_vector
        prev_erase_plane_info_var['position'] = new_position

def reset_plane_to_default(viewer, layer_name=data_name):
    print("Resetting plane to default position and normal.")
    if layer_name not in viewer.layers:
        print(f"Layer '{layer_name}' not found in the viewer.")
        return

    layer = viewer.layers[layer_name]
    
    if not hasattr(layer, 'plane'):
        print(f"Layer '{layer_name}' is not a plane layer.")
        return

    # Get the data shape
    data_shape = layer.data.shape
    center = np.array(data_shape) / 2

    layer.plane.position = center
    layer.plane.normal = (1, 0, 0)
    layer.visible = True

def reset_plane_view_to_default(viewer):
    global prev_camera_pos
    if viewer.dims.ndisplay == 3:
        reset_plane_to_default(viewer, data_name)
        if prev_camera_pos is not None:
            set_camera_view(viewer, prev_camera_pos)
        if label_3d_name in viewer.layers:
            update_label_from_3d_edit_layer(viewer)
            viewer.layers.remove(viewer.layers[label_3d_name])
        viewer.layers[main_label_name].visible = True

def shift_plane(layer, direction, padding_mode=False, padding=50):
    global plane_shift_status
    plane_shift_status = True
    if isinstance(layer, Image) and viewer.dims.ndisplay == 3 and layer.depiction == 'plane':
        # Get the current position and normal of the plane
        current_position = np.array(layer.plane.position)
        normal_vector = np.array(layer.plane.normal)
        
        # Normalize the normal vector
        normal_vector /= np.linalg.norm(normal_vector)
        
        if padding_mode:
            # Create the padding vector
            padding_vector = np.array([-padding, -padding, -padding])
            
            # Calculate the new position considering padding
            new_position = current_position - padding_vector
        else:
            # Simple shift without considering padding
            new_position = current_position + direction * normal_vector
        
        # Update the plane position
        layer.plane.position = tuple(new_position)
    elif viewer.dims.ndisplay == 2:
        # If in 2D mode, shift the slice by 1
        current_step = viewer.dims.current_step[0]
        if padding_mode:
            new_step = current_step + padding
        else:
            new_step = current_step + direction
        viewer.dims.set_current_step(0, new_step)
    else:
        print("Cannot shift: not in plane mode or 2D view")

def full_label_view(viewer):
    global prev_camera_pos
    if label_3d_name in viewer.layers:
            update_label_from_3d_edit_layer(viewer)
    if viewer.dims.ndisplay == 2:
        viewer.dims.ndisplay = 3
        for layer in viewer.layers:
            if layer.name != main_label_name:
                viewer.layers[layer.name].visible = False
            else:
                viewer.layers[layer.name].visible = True
                viewer.layers[layer.name].blending = 'opaque'
        if prev_camera_pos is not None:
            set_camera_view(viewer, prev_camera_pos)
    else:
        prev_camera_pos = get_current_camera_info(viewer)
        viewer.dims.ndisplay = 2
        for layer in viewer.layers:
            if layer.name != label_3d_name and layer.name != cc_preview_name:
                viewer.layers[layer.name].visible = True
            else:
                viewer.layers[layer.name].visible = False
            if layer.name == ink_label_name or layer.name == papyrus_label_name or layer.name == ff_name:
                viewer.layers[layer.name].blending = 'translucent'
        viewer.layers.selection.active = viewer.layers[main_label_name]
        viewer.layers[main_label_name].contour = 1
            
def switch_to_plane_view(viewer):
    global prev_camera_pos
    if label_3d_name in viewer.layers:
            update_label_from_3d_edit_layer(viewer)
# Switch to 3D mode
    if viewer.dims.ndisplay == 3:
        prev_camera_pos = get_current_camera_info(viewer)
        viewer.dims.ndisplay = 2
        for layer in viewer.layers:
            if layer.name == label_3d_name or layer.name == cc_preview_name:
                viewer.layers[layer.name].visible = False
            else:
                viewer.layers[layer.name].visible = True
            if layer.name == main_label_name:
                viewer.layers[layer.name].blending = 'translucent'
        viewer.layers.selection.active = viewer.layers[main_label_name]
        viewer.layers[main_label_name].contour = 1

    else:
        # Switch to 3D mode
        step_val = viewer.dims.current_step
        viewer.dims.ndisplay = 3
    
        # Prep layers visibility and blending
        for layer in viewer.layers:
            
            if layer.name != data_name and layer.name != ff_name and layer.name != main_label_name and layer.name != label_3d_name:
                viewer.layers[layer.name].visible = False
            elif layer.name == main_label_name:
                if label_3d_name in viewer.layers:
                    viewer.layers[label_3d_name].visible = True
                    viewer.layers[label_3d_name].blending = 'opaque'
                    viewer.layers[layer.name].visible = False
                else:
                    viewer.layers[layer.name].visible = True
                    viewer.layers[layer.name].blending = 'opaque'
            elif layer.name == data_name:
                # Change the depiction of `data_name` layer from volume to plane
                viewer.layers[layer.name].visible = True
                viewer.layers[layer.name].depiction = 'plane'
                viewer.layers[layer.name].plane.position = (step_val[0], 0, 0)
                viewer.layers[layer.name].affine = np.eye(3)  # Ensure the affine transform is identity for proper rendering
                viewer.layers[layer.name].blending = 'opaque'
                viewer.layers.selection.active = viewer.layers[layer.name]
        if prev_camera_pos is not None:
            set_camera_view(viewer, prev_camera_pos)

def update_label_from_3d_edit_layer(viewer):
    global previous_label_3d_data, manual_changes_mask

    if label_3d_name in viewer.layers and main_label_name in viewer.layers:
        existing_layer = viewer.layers[label_3d_name]

        if isinstance(existing_layer, napari.layers.Labels):
            # Calculate the manual changes mask
            if previous_label_3d_data is not None and previous_label_3d_data.shape == existing_layer.data.shape:
                manual_changes_mask = existing_layer.data != previous_label_3d_data
            else:
                manual_changes_mask = np.zeros_like(existing_layer.data, dtype=bool)
            
            # Apply the manual changes to the label_name layer
            viewer.layers[main_label_name].data[manual_changes_mask] = existing_layer.data[manual_changes_mask]
            
            # Refresh the viewer to immediately show the changes
            viewer.layers[main_label_name].refresh()

def setup_label_3d_layer(viewer, new_label_data, active_mode):
    global brush_size, pad_state, pad_amount

    # Remove the old label_3d_name layer if it exists
    visible_state = True
    if label_3d_name in viewer.layers:
        visible_state = viewer.layers[label_3d_name].visible
        viewer.layers.remove(viewer.layers[label_3d_name])
    
    # Add a new label layer with the updated data
    new_label_layer = viewer.add_labels(new_label_data, name=label_3d_name)
    if config.cube_config.use_custom_colormap:
        new_label_layer.colormap = get_direct_label_colormap()
    
    new_label_layer.visible = visible_state
    new_label_layer.blending = 'opaque'
    new_label_layer.opacity = 1
    new_label_layer.mode = active_mode
    new_label_layer.brush_size = brush_size

    if config.cube_config.smoother_labels:
        toggle_smooth_labels(viewer, new_label_layer, on=True)

    # Apply translation if pad_state is True
    if pad_state:
        new_label_layer.translate = [pad_amount, pad_amount, pad_amount]
    else:
        new_label_layer.translate = [0, 0, 0]

    setup_brush_size_listener(viewer, label_3d_name)

    # Refresh the viewer to immediately show the changes
    new_label_layer.refresh()

    return new_label_layer

def cut_label_at_plane(viewer, erase_mode=False, cut_side=True, prev_plane_info=None, recut=False):
    global previous_label_3d_data, manual_changes_mask, prev_plane_info_var, erase_slice_width, plane_shift_status, pad_state, pad_amount
    plane_shift_status = False
    data_plane = viewer.layers[data_name]
    if data_plane.depiction != 'plane':
        print("Please switch to plane mode.")
        return
    if viewer.layers.selection.active is not None:
        active_mode = viewer.layers.selection.active.mode
    else:
        active_mode = 'pan_zoom'
    if prev_plane_info is not None:
        position = prev_plane_info['position']
        normal = prev_plane_info['normal']
    elif erase_mode and prev_erase_plane_info_var is not None and not recut:
        position = prev_erase_plane_info_var['position']
        normal = prev_erase_plane_info_var['normal']
    else:
        position = np.array(data_plane.plane.position)
        normal = np.array(data_plane.plane.normal)
        prev_plane_info_var = {'position': position, 'normal': normal}
    viewer.layers[data_name].blending = 'opaque'

    # Adjust position based on pad_state
    offset = pad_amount if pad_state else 0
    adjusted_position = position - offset

    # Create a meshgrid for the label data coordinates
    label_shape = viewer.layers[main_label_name].data.shape
    z, y, x = np.meshgrid(np.arange(label_shape[0]),
                        np.arange(label_shape[1]),
                        np.arange(label_shape[2]),
                        indexing='ij')

    # Calculate the distance of each voxel from the plane
    distances = (x - adjusted_position[2]) * normal[2] + \
                (y - adjusted_position[1]) * normal[1] + \
                (z - adjusted_position[0]) * normal[0]

    # Update main_label_name layer from label_3d_name layer if it exists
    update_label_from_3d_edit_layer(viewer)

    # Create a copy of the label data and set all voxels between the viewer and the plane to 0
    new_label_data = viewer.layers[main_label_name].data.copy()
    if cut_side:
        new_label_data[distances > -2.5] = 0
        if erase_mode:
            new_label_data[distances < -erase_slice_width + 0.5] = 0
    else:
        new_label_data[distances < 2.5] = 0
        if erase_mode:
            new_label_data[distances > erase_slice_width + 0.5] = 0

    # Setup the label_3d_name layer
    setup_label_3d_layer(viewer, new_label_data, active_mode)

    # Store the current state of the label_3d_name layer for future comparison
    previous_label_3d_data = new_label_data.copy()

def plane_3d_erase_mode_shift_left(viewer):
    global erase_mode, prev_erase_plane_info_var, erase_slice_width
    overlap = erase_slice_width//5
    shift_prev_erase_plane(-erase_slice_width+overlap)
    shift_plane(viewer.layers[data_name], -erase_slice_width+overlap)
    if erase_mode and viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side)

def plane_3d_erase_mode_shift_right(viewer):
    global erase_mode, prev_erase_plane_info_var, erase_slice_width
    overlap = erase_slice_width//5
    shift_prev_erase_plane(erase_slice_width-overlap)
    shift_plane(viewer.layers[data_name], erase_slice_width-overlap)
    if erase_mode and viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side)

def shift_data_left_and_recut_3d_label(viewer):
    global erase_mode, cut_side
    shift_plane(viewer.layers[data_name], -1)
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side, recut=True)

def shift_data_left_fast_and_recut_3d_label(viewer):
    global erase_mode, cut_side, erase_slice_width
    if erase_mode:
        shift_plane(viewer.layers[data_name], -erase_slice_width)
    else:
        shift_plane(viewer.layers[data_name], -20)
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side, recut=True)

def shift_data_right_and_recut_3d_label(viewer):
    global erase_mode, cut_side
    shift_plane(viewer.layers[data_name], 1)
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side, recut=True)

def shift_data_right_fast_and_recut_3d_label(viewer):
    global erase_mode, cut_side, erase_slice_width
    if erase_mode:
        shift_plane(viewer.layers[data_name], erase_slice_width)
    else:
        shift_plane(viewer.layers[data_name], 20)
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side, recut=True)

# Global variables to track key states
left_key_pressed = False
right_key_pressed = False

# Define the functions to move left and right
def move_left(viewer, distance=1):
    shift_plane(viewer.layers[data_name], -distance)

def move_right(viewer, distance=1):
    shift_plane(viewer.layers[data_name], distance)

# Create timers for holding keys
left_timer = QTimer()
right_timer = QTimer()

# Connect the timers to the move functions
left_timer.timeout.connect(lambda: move_left(viewer))
right_timer.timeout.connect(lambda: move_right(viewer))

# Define the key press events
def shift_data_left(viewer):
    global left_key_pressed
    left_key_pressed = True
    move_left(viewer)  # Move immediately on key press
    if not left_timer.isActive():
        left_timer.start(500)  # Adjust the interval as needed

def shift_data_right(viewer):
    global right_key_pressed
    right_key_pressed = True
    move_right(viewer)  # Move immediately on key press
    if not right_timer.isActive():
        right_timer.start(500)  # Adjust the interval as needed

def shift_data_left_fast(viewer, distance=20):
    move_left(viewer, distance)

def shift_data_right_fast(viewer, distance=20):
    move_right(viewer, distance)

# Function to handle key release events
def handle_key_release(event):
    global left_key_pressed, right_key_pressed
    
    if event.key == 'a':
        left_key_pressed = False
    elif event.key == 'd':
        right_key_pressed = False
    
    # Check if keys are still pressed before stopping timers
    if not left_key_pressed and left_timer.isActive():
        left_timer.stop()
    if not right_key_pressed and right_timer.isActive():
        right_timer.stop()

# Function to handle key press events
def handle_key_press(event):
    if event.key == 'a':
        shift_data_left(viewer)
    elif event.key == 'd':
        shift_data_right(viewer)

# Connect the key events to the functions
viewer.window._qt_viewer.canvas.events.key_press.connect(handle_key_press)
viewer.window._qt_viewer.canvas.events.key_release.connect(handle_key_release)

def erase_mode_toggle(viewer):
    global eraser_size
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        viewer.layers[label_3d_name].mode = 'erase'
        viewer.layers.selection.active = viewer.layers[label_3d_name]
    elif viewer.dims.ndisplay == 3:
        viewer.layers[main_label_name].mode = 'erase'
        viewer.layers.selection.active = viewer.layers[main_label_name]
    elif viewer.dims.ndisplay == 2:
        viewer.layers[main_label_name].mode = 'erase'
        viewer.layers.selection.active = viewer.layers[main_label_name]

def plane_erase_3d_mode(viewer, switch=True):
    global erase_mode, cut_side, plane_shift_status, prev_erase_plane_info_var
    if not erase_mode:
        switch = False
        erase_mode = True
    if switch and not plane_shift_status:
        cut_side = not cut_side
    if viewer.dims.ndisplay == 3 and viewer.layers[data_name].depiction == 'plane' and viewer.layers[data_name].visible:
        position = np.array(viewer.layers[data_name].plane.position)
        normal = np.array(viewer.layers[data_name].plane.normal)
        prev_erase_plane_info_var = {'position': position, 'normal': normal}
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side)
        viewer.layers[main_label_name].visible = False

def move_mode(viewer):
    if viewer.dims.ndisplay == 3 and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        viewer.layers[label_3d_name].mode = 'pan_zoom'
    else:
        viewer.layers[main_label_name].mode = 'pan_zoom'

def cut_label_at_oblique_plane(viewer, switch=True, prev_plane_info=None):
    global erase_mode, cut_side, plane_shift_status
    if erase_mode:
        switch = False
        erase_mode = False
    if switch and not plane_shift_status:
        cut_side = not cut_side
    if viewer.dims.ndisplay == 3 and viewer.layers[data_name].depiction == 'plane' and viewer.layers[data_name].visible:
        cut_label_at_plane(viewer, erase_mode=False, cut_side=cut_side, prev_plane_info=prev_plane_info)
        viewer.layers[label_3d_name].visible = True
        viewer.layers[main_label_name].visible = False

def connected_components(viewer, preview=False, cc_layer_name=main_label_name):
    global erase_mode, cut_side

    if not preview and viewer.layers[main_label_name].data is not None and data_manager.original_ink_pred_data is not None and data_manager.label_data is not None:
        cc_layer_name = select_from_list_popup("Connected Components", "Select the layer to apply connected components to", [papyrus_label_name, ink_label_name])
    if not preview:
        msg = "DANGER Are you sure you want to run connected components? This operation cannot be undone and removes the undo queue. Consider saving first. \n\nIF YOU HAVE DILATED SEPERATED LABELS AND THEY NOW TOUCH, THEY WILL BE COMBINED."
        response = confirm_popup(msg)
        if response != QMessageBox.Yes:
                return 

    #apply any changes to the label_3d_name layer to the labels layer
    if label_3d_name in viewer.layers:
        update_label_from_3d_edit_layer(viewer)

    if  preview and label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cc_data = viewer.layers[label_3d_name].data.copy()
    else:
        cc_data = viewer.layers[cc_layer_name].data.copy()
    print(np.sum(cc_data > 0))

    cc_result = label_foreground_structures_napari(cc_data, min_size=config.cube_config.cc_min_size)
    print(cc_result.shape)
    print(np.sum(cc_result > 0))
    if preview:
        if cc_preview_name in viewer.layers:
            viewer.layers[cc_preview_name].data = cc_result
        else:
            viewer.add_labels(cc_result, name=cc_preview_name)
        viewer.layers[cc_preview_name].visible = True
        if config.cube_config.use_custom_colormap:
            viewer.layers[cc_preview_name].colormap = get_direct_label_colormap()
        viewer.layers[cc_preview_name].editable = False
        viewer.layers[main_label_name].visible = False
        if label_3d_name in viewer.layers:
            viewer.layers[label_3d_name].visible = False
    else:
        viewer.layers[cc_layer_name].data = cc_result
    
    if label_3d_name in viewer.layers and viewer.layers[label_3d_name].visible:
        cut_label_at_plane(viewer, erase_mode=erase_mode, cut_side=cut_side)
    msg = 'connected components finished'
    if not preview:
        show_popup(msg)
    viewer.status = msg
    print(msg)

def toggle_contextual_view(viewer):
    global pad_state, prev_erase_plane_info_var

    # Define the translation offset
    offset = np.array([pad_amount, pad_amount, pad_amount])

    if pad_state:
        # Remove the offset
        for layer in viewer.layers:
            if layer.name != data_name:
                layer.translate -= offset
        
        # Reset the data layer to the original (unpadded) data
        viewer.layers[data_name].data = data_manager.raw_data
        shift_plane(viewer.layers[data_name], 0, padding_mode=True, padding=-pad_amount)
        # Adjust the camera position
        viewer.camera.center -= offset
        # Adjust prev_erase_plane_info_var
        if prev_erase_plane_info_var is not None:
            prev_erase_plane_info_var['position'] -= offset

        pad_state = False
    else:
        # Add the offset
        for layer in viewer.layers:
            if layer.name != data_name:
                layer.translate += offset
        
        # Set the data layer to the padded data
        viewer.layers[data_name].data = data_manager.padded_raw_data
        shift_plane(viewer.layers[data_name], 0, padding_mode=True, padding=pad_amount)
        # Adjust the camera position
        viewer.camera.center += offset
        # Adjust prev_erase_plane_info_var
        if prev_erase_plane_info_var is not None:
            prev_erase_plane_info_var['position'] += offset

        pad_state = True

@thread_worker
def save_labels_worker(viewer, z, y, x, papyrus_labels, ink_labels, seg_mesh_labels):
    msg = 'save labels'
    viewer.status = msg
    print(msg)

    async def save_all():
        tasks = []

        # Save ink prediction data if it exists
        if ink_labels is not None:
            if ink_labels.shape[0] != config.cube_config.chunk_size:
                ink_labels_unpadded = unpad_array(ink_labels, config.cube_config.chunk_size)
            else:
                ink_labels_unpadded = ink_labels
            tasks.append(data_manager.save_label_data_async(z, y, x, ink_labels_unpadded, 'ink'))

        # Save papyrus label data
        if papyrus_labels is not None:
            if papyrus_labels.shape[0] != config.cube_config.chunk_size:
                papyrus_labels_unpadded = unpad_array(papyrus_labels, config.cube_config.chunk_size)
            else:
                papyrus_labels_unpadded = papyrus_labels
            tasks.append(data_manager.save_label_data_async(z, y, x, papyrus_labels_unpadded, 'vol'))

        # Save segmentation mesh data
        if seg_mesh_labels is not None:
            tasks.append(data_manager.save_seg_mesh_data_async(z, y, x, seg_mesh_labels))

        # Save raw data if it hasn't been saved before
        tasks.append(data_manager.save_raw_data_async(z, y, x))

        # Wait for all save tasks to complete
        await asyncio.gather(*tasks)

    # Run the async function in the thread
    asyncio.run(save_all())

    return data_manager.get_output_path(z, y, x)

def save_labels(viewer, z=None, y=None, x=None, should_show_popup=True, papyrus_labels=None, ink_labels=None, seg_mesh_labels=None):
    if data_manager.is_saving:
        show_popup("A save operation is in progress. Please wait a few seconds before navigating to a new cube.")
        return False
    if x is None:
        x = config.cube_config.x
    if y is None:
        y = config.cube_config.y
    if z is None:
        z = config.cube_config.z
    if label_3d_name in viewer.layers:
        update_label_from_3d_edit_layer(viewer)
    if papyrus_labels is None and papyrus_label_name in viewer.layers:
        papyrus_labels = viewer.layers[papyrus_label_name].data
    if ink_labels is None and ink_label_name in viewer.layers:
        ink_labels = viewer.layers[ink_label_name].data
    if seg_mesh_labels is None and seg_mesh_name in viewer.layers:
        seg_mesh_labels = viewer.layers[seg_mesh_name].data

    worker = save_labels_worker(viewer, z, y, x, papyrus_labels, ink_labels, seg_mesh_labels)
    
    if should_show_popup:
        worker.started.connect(lambda: show_popup("Saving has started. You will be notified when it's complete."))
        worker.finished.connect(lambda: show_popup("Saving complete"))
    
    worker.returned.connect(lambda output_path: print(f"Layers saved to {output_path}"))
    worker.errored.connect(lambda error: print(f"Error during saving: {error}"))

    worker.start()

def reset_segmentation_mesh(viewer):
    global data_manager, config

    # Reload the segmentation mesh data from b2nd files
    current_directory = os.getcwd()
    root_directory = f"{current_directory}/data/manual_sheet_segmentation/{config.cube_config.scroll_name}"
    new_seg_mesh_data = data_manager.process_b2nd_files(
        root_directory, 
        config.cube_config.z_num, 
        config.cube_config.y_num, 
        config.cube_config.x_num, 
        config.cube_config.chunk_size, 
        config.cube_config.voxelized_mesh_pad_amount
    )

    # Update the data_manager with the new segmentation mesh data
    data_manager.voxelized_segmentation_mesh_data = new_seg_mesh_data

    # Update the viewer layer
    if seg_mesh_name in viewer.layers:
        viewer.layers[seg_mesh_name].data = new_seg_mesh_data
    else:
        seg_mesh_layer = viewer.add_labels(new_seg_mesh_data, name=seg_mesh_name)
    
    offset = np.array([config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount])
    viewer.layers[seg_mesh_name].translate = -offset
    viewer.layers[seg_mesh_name].opacity = 1
    if config.cube_config.use_custom_colormap:
        viewer.layers[seg_mesh_name].colormap = get_direct_label_colormap()

    print("Segmentation mesh has been reset to the original b2nd data.")
    show_info("Segmentation mesh has been reset.")

def connected_components_preview(viewer):
    if cc_preview_name in viewer.layers and viewer.layers[cc_preview_name].visible:
        viewer.layers[cc_preview_name].visible = False
        if label_3d_name in viewer.layers:
            viewer.layers[label_3d_name].visible = True
        else:
            viewer.layers[main_label_name].visible = True
    else:
        connected_components(viewer, preview=True)
    
def bind_hotkeys(viewer, hotkey_config, module=None, overwrite=True):
    if module is None:
        module = sys.modules['__main__']  # Get the main module
    
    for func_name, keys in hotkey_config.hotkey_config.items():
        # Skip if keys is None, an empty string, or an empty list
        if keys is None or keys == "" or (isinstance(keys, list) and len(keys) == 0):
            print(f"Warning: No key binding specified for function '{func_name}'. Skipping.")
            continue

        # First, try to find the function in the viewer
        if hasattr(viewer, func_name):
            func = getattr(viewer, func_name)
        # If not in viewer, try to find it in the main module
        elif hasattr(module, func_name):
            func = getattr(module, func_name)
        else:
            print(f"Warning: Function '{func_name}' not found. Skipping.")
            continue

        if isinstance(keys, list):
            for key in keys:
                if key:  # Only bind if key is not an empty string
                    try:
                        viewer.bind_key(key, func, overwrite=overwrite)
                    except (ValueError, TypeError) as e:
                        print(f"Error binding key '{key}' to function '{func_name}': {str(e)}")
        elif keys:  # Only bind if keys is not an empty string
            try:
                viewer.bind_key(keys, func, overwrite=overwrite)
            except (ValueError, TypeError) as e:
                print(f"Error binding key '{keys}' to function '{func_name}': {str(e)}")

def cleanup_labels(viewer):
    viewer.layers[main_label_name].data = filter_and_reassign_labels(viewer.layers[main_label_name].data, config.cube_config.cc_min_size)
    show_popup("Labels clean up finished")


def update_and_reload_data(viewer, data_manager, config, new_z, new_y, new_x, new_chunk_size=None):
    if data_manager.is_saving:
        show_popup("A save operation is in progress. Please wait a few seconds before navigating to a new cube.")
        return False
    new_z = str(new_z).zfill(5)
    new_y = str(new_y).zfill(5)
    new_x = str(new_x).zfill(5)

    # Save the current labels, including segmentation mesh, before updating the coordinates
    if label_3d_name in viewer.layers:
        update_label_from_3d_edit_layer(viewer)
    papyrus_labels = viewer.layers[papyrus_label_name].data if papyrus_label_name in viewer.layers else None
    ink_labels = viewer.layers[ink_label_name].data if ink_label_name in viewer.layers else None
    seg_mesh_labels = viewer.layers[seg_mesh_name].data if seg_mesh_name in viewer.layers else None
    
    save_labels(viewer, config.cube_config.z, config.cube_config.y, config.cube_config.x, 
                should_show_popup=False, papyrus_labels=papyrus_labels, ink_labels=ink_labels, seg_mesh_labels=seg_mesh_labels)
    
    config.cube_config.update_coordinates(new_z, new_y, new_x)

    if new_chunk_size is not None:
        config.cube_config.chunk_size = new_chunk_size
    data_manager.reload_data()

    # Update the layers in the viewer
    viewer.layers[data_name].data = data_manager.raw_data
    if data_manager.label_data is not None:
        if papyrus_label_name in viewer.layers:
            viewer.layers[papyrus_label_name].data = data_manager.label_data
        else:
            viewer.add_labels(data_manager.label_data, name=papyrus_label_name)
    elif papyrus_label_name in viewer.layers:
        viewer.layers.remove(papyrus_label_name)

    if data_manager.original_ink_pred_data is not None:
        if ink_label_name in viewer.layers:
            viewer.layers[ink_label_name].data = data_manager.original_ink_pred_data
        else:
            viewer.add_labels(data_manager.original_ink_pred_data, name=ink_label_name)
    elif ink_label_name in viewer.layers:
            viewer.layers.remove(ink_label_name)

    if data_manager.original_label_data is not None:
        if "Original Semantic Mask" in viewer.layers:
            viewer.layers["Original Semantic Mask"].data = data_manager.original_label_data
        else:
            viewer.add_labels(data_manager.original_label_data, name="Original Semantic Mask")
    elif "Original Semantic Mask" in viewer.layers:
        viewer.layers.remove("Original Semantic Mask")

    if data_manager.voxelized_segmentation_mesh_data is not None:
        if seg_mesh_name in viewer.layers:
            viewer.layers[seg_mesh_name].data = data_manager.voxelized_segmentation_mesh_data
        else:
            seg_mesh_layer = viewer.add_labels(data_manager.voxelized_segmentation_mesh_data, name=seg_mesh_name)
        offset = np.array([config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount, config.cube_config.voxelized_mesh_pad_amount])
        viewer.layers[seg_mesh_name].translate = -offset
        viewer.layers[seg_mesh_name].opacity = 1
        if config.cube_config.use_custom_colormap:
            viewer.layers[seg_mesh_name].colormap = get_direct_label_colormap()
    elif seg_mesh_name in viewer.layers:
        viewer.layers.remove(seg_mesh_name)

    if label_3d_name in viewer.layers:
        viewer.layers.remove(label_3d_name)
    if cc_preview_name in viewer.layers:
        viewer.layers.remove(cc_preview_name)
    if ff_name in viewer.layers:
        viewer.layers.remove(ff_name)

    if main_label_name in viewer.layers:
        viewer.layers.selection.active = viewer.layers[main_label_name]
        viewer.layers[main_label_name].visible = True
        viewer.layers[main_label_name].contour = 1
        viewer.layers[main_label_name].opacity = 1
        viewer.layers[main_label_name].blending = 'opaque'
        if config.cube_config.use_custom_colormap:
            viewer.layers[main_label_name].colormap = get_direct_label_colormap()
    elif data_name in viewer.layers:
        viewer.layers.selection.active = viewer.layers[data_name]

    return True

# Create a dictionary of functions to pass to the GUI
functions_dict = {
    'erode_labels': erode_labels,
    'dilate_labels': dilate_labels,
    'full_label_view': full_label_view,
    'switch_to_plane': switch_to_plane_view,
    'toggle_contextual_view': toggle_contextual_view,
    'cut_label_at_oblique_plane': cut_label_at_oblique_plane,
    'connected_components': connected_components,
    'color_semantic_mask': color_semantic_mask,
    'save_labels': save_labels,
    'update_and_reload_data': lambda z, y, x: update_and_reload_data(viewer, data_manager, config, z, y, x),
    'move_seg_mesh_label': lambda viewer, dz, dy, dx, move_all: move_seg_mesh_label(viewer, dz, dy, dx, move_all),
    'reset_segmentation_mesh': reset_segmentation_mesh,
    'morphological_tunnel_fill': morphological_tunnel_fill,
}

def update_global_erase_slice_width(value):
    global erase_slice_width
    erase_slice_width = value
    print(f"Global erase width updated to: {erase_slice_width}")

# Create the GUI
seg_mesh_exists = seg_mesh_name in viewer.layers # Check if the segmentation mesh layer exists
gui = VesuviusGUI(viewer, functions_dict, update_global_erase_slice_width, config, config.cube_config.main_label_layer_name, seg_mesh_exists)
gui.setup_napari_defaults(main_label_name)
if papyrus_label_name in viewer.layers and config.cube_config.use_custom_colormap:
    papyrus_label_layer.colormap = get_direct_label_colormap()
if seg_mesh_name in viewer.layers and config.cube_config.use_custom_colormap:
    seg_mesh_layer.colormap = get_direct_label_colormap()

bind_hotkeys(viewer, config.hotkey_config)
set_camera_view(viewer)
setup_brush_size_listener(viewer, main_label_name)

try:
    viewer.window.add_plugin_dock_widget(
        plugin_name="napari-threedee", widget_name="render plane manipulator"
    )
except Exception as e:
    print(f"Error adding render plane manipulator widget: {str(e)}")

try:
    viewer.window.add_plugin_dock_widget(
        "napari-threedee", widget_name="label annotator"
    )
except Exception as e:
    print(f"Error adding label annotator widget: {str(e)}")

if config.cube_config.smoother_labels:
    if papyrus_label_name in viewer.layers:
        toggle_smooth_labels(viewer, viewer.layers[papyrus_label_name], on=True)
    if ink_label_name in viewer.layers:
        toggle_smooth_labels(viewer, viewer.layers[ink_label_name], on=True)
    if seg_mesh_name in viewer.layers:
        toggle_smooth_labels(viewer, viewer.layers[seg_mesh_name], on=True)
    if "Original Semantic Mask" in viewer.layers:
        toggle_smooth_labels(viewer, viewer.layers["Original Semantic Mask"], on=True)

viewer.window.add_dock_widget(toggle_smooth_labels)

# Start the Napari event loop
napari.run()