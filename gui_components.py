from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel, 
                             QVBoxLayout, QScrollArea, QSizePolicy, QSpinBox, QColorDialog,
                             QLineEdit, QGridLayout, QComboBox)
from PyQt5.QtCore import Qt
from helper import *

MIN_BUTTON_WIDTH = 150

def pick_color(viewer):
    # Open a color picker dialog
    color = QColorDialog.getColor()

    if color.isValid():
        # Convert QColor to a tuple of RGBA values normalized to [0, 1]
        color_tuple = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
        # Set the background color of the canvas
        viewer.window._qt_viewer.canvas.bgcolor = color_tuple

class MoveSegMeshWidget(QWidget):
    def __init__(self, viewer, move_function, reset_function):
        super().__init__()
        self.viewer = viewer
        self.move_function = move_function
        self.reset_function = reset_function
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Dropdown menu for move type
        self.move_type_combo = QComboBox()
        self.move_type_combo.addItems(["Selected Label", "All Labels"])
        layout.addWidget(self.move_type_combo)

        # dz, dy, dx inputs
        for axis in ['dz', 'dy', 'dx']:
            axis_layout = QHBoxLayout()
            axis_layout.addWidget(QLabel(f"{axis}:"))
            spinbox = QSpinBox()
            spinbox.setRange(-100, 100)  # Adjust range as needed
            spinbox.setValue(0)
            setattr(self, f"{axis}_spinbox", spinbox)
            axis_layout.addWidget(spinbox)
            layout.addLayout(axis_layout)

        # Move button
        move_button = QPushButton("Move Label(s)")
        move_button.clicked.connect(self.move_label)
        layout.addWidget(move_button)

        # Reset button
        reset_button = QPushButton("Reset Segmentation Mesh")
        reset_button.clicked.connect(self.confirm_reset_mesh)
        layout.addWidget(reset_button)

    def move_label(self):
        dz = self.dz_spinbox.value()
        dy = self.dy_spinbox.value()
        dx = self.dx_spinbox.value()
        move_all = self.move_type_combo.currentText() == "All Labels"
        self.move_function(self.viewer, dz, dy, dx, move_all)

    def confirm_reset_mesh(self):
        message = "Are you sure you want to reset the segmentation mesh?\n\nThis action will reload the mesh from the b2nd files and cannot be undone."
        if confirm_popup(message):
            self.reset_function(self.viewer)

class ZYXNavigationWidget(QWidget):
    def __init__(self, config, update_function, viewer):
        super().__init__()
        self.config = config
        self.update_function = update_function
        self.viewer = viewer
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ZYX input
        zyx_layout = QHBoxLayout()
        zyx_layout.addWidget(QLabel("ZYX:"))
        self.zyx_input = QLineEdit(self.config.cube_config.zyx)
        self.zyx_input.setPlaceholderText("ZZZZZ_YYYYY_XXXXX")
        self.zyx_input.returnPressed.connect(self.update_zyx)
        zyx_layout.addWidget(self.zyx_input)
        layout.addLayout(zyx_layout)

        # Navigation buttons
        axes = ['Z', 'Y', 'X']
        for axis in axes:
            axis_layout = QHBoxLayout()
            axis_layout.addWidget(QLabel(f"{axis}:"))
            prev_button = QPushButton(f"Previous")
            next_button = QPushButton(f"Next")
            prev_button.clicked.connect(lambda checked, a=axis, d=-1: self.navigate_cube(a, d))
            next_button.clicked.connect(lambda checked, a=axis, d=1: self.navigate_cube(a, d))
            axis_layout.addWidget(prev_button)
            axis_layout.addWidget(next_button)
            layout.addLayout(axis_layout)

    def update_zyx(self, new_zyx=None):
        
        if new_zyx is None:
            new_zyx = self.zyx_input.text()
        if self.validate_zyx(new_zyx):
            z, y, x = new_zyx.split('_')
            
            #only update coord text if update is successful
            if self.update_function(z, y, x):
                self.zyx_input.setText(new_zyx)
        else:
            show_popup("Invalid ZYX format. Please use ZZZZZ_YYYYY_XXXXX.")

    def navigate_cube(self, axis, direction):
        current_z = int(self.config.cube_config.z)
        current_y = int(self.config.cube_config.y)
        current_x = int(self.config.cube_config.x)
        chunk_size = self.config.cube_config.chunk_size

        if axis == 'Z':
            new_z = current_z + (direction * chunk_size)
            if new_z < 0:
                show_popup("Cannot navigate to negative Z values.")
                return
            new_zyx = f"{new_z:05d}_{current_y:05d}_{current_x:05d}"
        elif axis == 'Y':
            new_y = current_y + (direction * chunk_size)
            if new_y < 0:
                show_popup("Cannot navigate to negative Y values.")
                return
            new_zyx = f"{current_z:05d}_{new_y:05d}_{current_x:05d}"
        elif axis == 'X':
            new_x = current_x + (direction * chunk_size)
            if new_x < 0:
                show_popup("Cannot navigate to negative X values.")
                return
            new_zyx = f"{current_z:05d}_{current_y:05d}_{new_x:05d}"

        # self.zyx_input.setText(new_zyx)
        self.update_zyx(new_zyx)

    @staticmethod
    def validate_zyx(zyx):
        parts = zyx.split('_')
        return len(parts) == 3 and all(len(part) == 5 and part.isdigit() for part in parts)

# Create a custom widget with a button to open the color picker
class ColorPickerWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a button to open the color picker
        button = QPushButton("Pick Background Color")
        button.clicked.connect(lambda: pick_color(self.viewer))
        layout.addWidget(button)

class CustomButtonWidget(QWidget):
    def __init__(self, button_text, hotkey, callback_function):
        super().__init__()
        layout = QHBoxLayout()
        
        # Create button
        button = QPushButton(button_text)
        button.setMinimumWidth(MIN_BUTTON_WIDTH)
        button.clicked.connect(callback_function)
        
        # Create hotkey label
        hotkey_label = QLabel(hotkey)
        hotkey_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add button and label to layout
        layout.addWidget(button)
        layout.addWidget(hotkey_label)
        
        self.setLayout(layout)

class FillHolesWidget(QWidget):
    def __init__(self, fill_holes_function):
        super().__init__()
        self.fill_holes_function = fill_holes_function
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Closing iterations input
        closing_iterations_layout = QHBoxLayout()
        closing_iterations_layout.addWidget(QLabel("Closing Iterations:"))
        self.closing_iterations_spinbox = QSpinBox()
        self.closing_iterations_spinbox.setMinimum(1)
        self.closing_iterations_spinbox.setMaximum(10)
        self.closing_iterations_spinbox.setValue(1)
        closing_iterations_layout.addWidget(self.closing_iterations_spinbox)
        layout.addLayout(closing_iterations_layout)

        # Fill Holes button
        self.fill_holes_button = QPushButton("Fill Holes")
        self.fill_holes_button.clicked.connect(self.fill_holes)
        layout.addWidget(self.fill_holes_button)

    def fill_holes(self):
        iterations = self.closing_iterations_spinbox.value()
        self.fill_holes_function(iterations)

class VesuviusGUI:
    def __init__(self, viewer, functions_dict, update_global_erase_slice_width, config, main_label_layer_name='Papyrus Labels', seg_mesh_exists=False):
        self.viewer = viewer
        self.functions = functions_dict
        self.update_global_erase_slice_width = update_global_erase_slice_width
        self.erase_slice_width = 30
        self.config = config
        self.main_label_layer_name = main_label_layer_name
        self.move_seg_mesh_widget = None 
        self.setup_gui(seg_mesh_exists)

    def get_key_string(self, func):
        if not hasattr(self.config.hotkey_config, func):
            print(f'Function {func} not found in hotkey config')
            return ''
        keys = getattr(self.config.hotkey_config, func)
        if isinstance(keys, list):
            return ' or '.join(str(key) for key in keys if key)
        return str(keys) if keys else ''

    def create_button_container(self):
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Set smaller margins for the layout
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Set smaller spacing between widgets
        layout.setSpacing(8)
        
        # Add dropdown menu for erode/dilate label selection
        self.label_selection_combo = QComboBox()
        self.label_selection_combo.addItems(["Erode/Dilate All Labels", "Erode/Dilate Selected Label", "Erode/Dilate All Labels No Warning", "Erode/Dilate Selected Label No Warning"])
        layout.addWidget(self.label_selection_combo)
        
        buttons = [self.dilate_button, self.erode_button, self.full_view_button, self.plane_cut_button, 
                self.cut_plane_button, self.padding_button, self.components_button, self.color_semantic_label_button, self.save_button]
        
        for button in buttons:
            layout.addWidget(button)
            
            # If it's a CustomButtonWidget, adjust its internal layout
            if isinstance(button, CustomButtonWidget):
                button_layout = button.layout()
                button_layout.setSpacing(2)
                button_layout.setContentsMargins(2, 2, 2, 2)
                
                # Adjust the size policy of the button and label
                button.findChild(QPushButton).setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                button.findChild(QLabel).setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Add a stretch at the end to push all widgets to the top
        layout.addStretch(1)
        
        return container

    def create_erase_width_input(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Erase slice Width:"))
        self.erase_slice_width_spinbox = QSpinBox()
        self.erase_slice_width_spinbox.setMinimum(1)
        self.erase_slice_width_spinbox.setMaximum(100)
        self.erase_slice_width_spinbox.setValue(self.erase_slice_width)
        self.erase_slice_width_spinbox.valueChanged.connect(self.update_erase_slice_width)
        layout.addWidget(self.erase_slice_width_spinbox)
        return layout
    

    def create_instruction_scroll_area(self):
        text_container = QWidget()
        text_layout = QVBoxLayout()
        text_container.setLayout(text_layout)
        instruction_label = QLabel(self.get_instruction_text())
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        text_layout.addWidget(instruction_label)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(text_container)
        return scroll_area

    def setup_gui(self, seg_mesh_exists=False):
        # Create custom button widgets
        if seg_mesh_exists:
            self.move_seg_mesh_widget = MoveSegMeshWidget(self.viewer, self.functions['move_seg_mesh_label'], self.functions['reset_segmentation_mesh'])
            self.viewer.window.add_dock_widget(self.move_seg_mesh_widget, area='left', name='Move Seg Mesh')
            self.color_semantic_label_button = CustomButtonWidget("Color Semantic Mask", '', self.color_semantic_mask)
        else: 
            self.color_semantic_label_button = None
            
        self.zyx_widget = ZYXNavigationWidget(self.config, self.functions['update_and_reload_data'], self.viewer)
        self.viewer.window.add_dock_widget(self.zyx_widget, area='left', name='ZYX Navigation')

        self.dilate_button = CustomButtonWidget("Dilate Labels", self.get_key_string('dilate_labels'), self.dilate_labels_gui)
        self.erode_button = CustomButtonWidget("Erode Labels", self.get_key_string('erode_labels'), self.erode_labels_gui)
        self.full_view_button = CustomButtonWidget("Toggle Full Label View", self.get_key_string('full_label_view'), self.toggle_full_label_view)
        self.plane_cut_button = CustomButtonWidget("Toggle 3D Plane Cut View", self.get_key_string('switch_to_plane_view'), self.toggle_3D_plane_cut_view)
        self.padding_button = CustomButtonWidget("Toggle Padding Context", self.get_key_string('toggle_contextual_view'), self.toggle_padding_context)
        self.cut_plane_button = CustomButtonWidget("Cut Label at Plane", self.get_key_string('cut_label_at_oblique_plane'), self.cut_label_at_plane_gui)
        self.components_button = CustomButtonWidget("Connected Components", self.get_key_string('connected_components'), self.run_connected_components)
        self.save_button = CustomButtonWidget("Save Labels", self.get_key_string('save_labels'), self.save_labels_button)
        color_picker_widget = ColorPickerWidget(self.viewer)

        button_container = self.create_button_container()
        erase_width_layout = self.create_erase_width_input()

        main_container = QWidget()
        main_layout = QVBoxLayout()
        main_container.setLayout(main_layout)
        main_container.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        main_layout.addWidget(button_container)
        main_layout.addLayout(erase_width_layout)
        main_layout.addWidget(color_picker_widget)

        self.viewer.window.add_dock_widget(main_container, area='right')

        instruction_scroll_area = self.create_instruction_scroll_area()
        self.viewer.window.add_dock_widget(instruction_scroll_area, area='right')
        
        self.fill_holes_widget = FillHolesWidget(lambda iterations: self.functions['fill_holes'](self.viewer, iterations))
        self.viewer.window.add_dock_widget(self.fill_holes_widget, area='right', name='Fill Holes')

    def update_erase_slice_width(self, value):
        self.erase_slice_width = value
        self.update_global_erase_slice_width(value)
        print(f"Erase width updated to: {self.erase_slice_width}")

    def setup_napari_defaults(self, main_label_layer_name='Papyrus Labels'):
        viewer = self.viewer
        data_name = 'Data'
        ff_name = 'flood_fill_layer'
        label_3d_name = '3D Label Edit Layer'
        self.viewer.axes.visible = True
        self.viewer.dims.ndisplay = 3   
        main_label_layer = self.viewer.layers[main_label_layer_name]
        main_label_layer.n_edit_dimensions = 3
        main_label_layer.opacity = 1
        main_label_layer.contour = 1
        main_label_layer.brush_size = 4
        self.viewer.theme = 'light'
        self.viewer.window._qt_viewer.canvas.bgcolor = (0.68, 0.85, 0.90, 1.0)
        main_label_layer.shape = 'square'
        self.viewer.layers.selection.active = main_label_layer
        # Prep layers visibility and blending
        step_val = viewer.dims.current_step
        for layer in viewer.layers:
            
            if layer.name != data_name and layer.name != ff_name and layer.name != main_label_layer_name and layer.name != label_3d_name:
                viewer.layers[layer.name].visible = False
                viewer.layers[layer.name].blending = 'opaque'
            elif layer.name == main_label_layer_name:
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
    
    # Define button callback methods
    def dilate_labels_gui(self):
        self.functions['dilate_labels'](self.viewer)

    def erode_labels_gui(self):
        self.functions['erode_labels'](self.viewer)

    def fill_holes(self):
        iterations = self.closing_iterations_spinbox.value()
        self.functions['fill_holes'](self.viewer, iterations)

    def toggle_full_label_view(self):
        self.functions['full_label_view'](self.viewer)

    def toggle_3D_plane_cut_view(self):
        self.functions['switch_to_plane_view'](self.viewer)

    def toggle_padding_context(self):
        self.functions['toggle_contextual_view'](self.viewer)

    def cut_label_at_plane_gui(self):
        self.functions['cut_label_at_oblique_plane'](self.viewer)

    def run_connected_components(self):
        self.functions['connected_components'](self.viewer)

    def color_semantic_mask(self):
        self.functions['color_semantic_mask'](self.viewer)

    def save_labels_button(self):
        self.functions['save_labels'](self.viewer, self.config.cube_config.z, self.config.cube_config.y, self.config.cube_config.x)

    def get_instruction_text(self):
        def key_to_string(key):
            key_map = {
                'Left': 'Left Arrow',
                'Right': 'Right Arrow',
                'Up': 'Up Arrow',
                'Down': 'Down Arrow',
                'Shift-Left': 'Shift-Left Arrow',
                'Shift-Right': 'Shift-Right Arrow',
                'Shift-Up': 'Shift-Up Arrow',
                'Shift-Down': 'Shift-Down Arrow',
            }
            return key_map.get(key, key)

        def get_key_string(func):
            if not hasattr(self.config.hotkey_config, func):
                return 'unassigned'
            keys = getattr(self.config.hotkey_config, func)
            if not keys:
                return 'unassigned'
            if isinstance(keys, list):
                return ' or '.join(f'<b>{key_to_string(key)}</b>' for key in keys if key)
            return f'<b>{key_to_string(keys)}</b>'

        instruction_text = f"""
        <b>Custom Napari Keybinds:</b><br>
        - {get_key_string('toggle_labels_visibility')} to toggle label visibility<br>
        - {get_key_string('toggle_data_visibility')} to toggle data visibility<br>
        - {get_key_string('reset_plane_to_default')} to reset data plane to default position<br>
        - {get_key_string('cut_label_at_oblique_plane')} to cut label at 3D plane location<br>
        - {get_key_string('switch_to_data_layer')} to switch active layer to data layer<br>
        - {get_key_string('full_label_view')} to toggle full 3D label view<br>
        - {get_key_string('switch_to_plane_view')} to toggle 3D plane cut view layers<br>
        - {get_key_string('erase_mode_toggle')} to switch to erase mode<br>
        - {get_key_string('move_mode')} to switch to pan & zoom mode<br>
        - {get_key_string('plane_erase_3d_mode')} to toggle 3D plane precision erase mode<br>
        - <b>shift + click</b> to move the 3D volume plane quickly<br>
        - <b>shift + right click + drag up or down</b> to 'fisheye' the view<br>\n
        - {get_key_string('erode_labels')} to erode labels 1 iteration<br>
        - {get_key_string('dilate_labels')} to dilate labels 1 iteration<br>
        - {get_key_string('toggle_contextual_view')} to toggle context padding data<br>
        - {get_key_string('connected_components')} to run connected components analysis and relabel<br>
        - {get_key_string('flood_fill')} for standard flood fill<br>
        - {get_key_string('large_flood_fill')} for larger flood fill<br>
        - {get_key_string('save_labels')} to save data & labels as nrrd files<br>\n
        - {get_key_string('decrease_brush_size')} or '[' to decrease brush size<br>
        - {get_key_string('increase_brush_size')} or ']' to increase brush size<br>
        - {get_key_string('toggle_show_selected_label')} to toggle show selected label<br>
        """
        return instruction_text