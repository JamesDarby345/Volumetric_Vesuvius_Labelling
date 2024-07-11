from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel, 
                             QVBoxLayout, QScrollArea, QSizePolicy, QSpinBox, QColorDialog)
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

class VesuviusGUI:
    def __init__(self, viewer, functions_dict, update_global_erase_slice_width, config, main_label_layer_name='Papyrus Labels'):
        self.viewer = viewer
        self.functions = functions_dict  # Store the functions dictionary
        self.update_global_erase_slice_width = update_global_erase_slice_width
        self.erase_slice_width = 30
        self.config = config
        self.setup_gui()
        self.main_label_layer_name = main_label_layer_name

    def get_key_string(self, func):
        keys = self.config.get(func, [])
        if isinstance(keys, list):
            return ' or '.join(keys)
        return str(keys)

    def create_button_container(self):
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Set smaller margins for the layout
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Set smaller spacing between widgets
        layout.setSpacing(8)
        
        buttons = [self.dilate_button, self.erode_button, self.full_view_button, self.plane_cut_button, 
                self.cut_plane_button, self.padding_button, self.components_button, self.save_button]
        
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
        instruction_label = QLabel(self.get_instruction_text(self.config))
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        text_layout.addWidget(instruction_label)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(text_container)
        return scroll_area

    def setup_gui(self):
        # Create custom button widgets
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

    def save_labels_button(self):
        self.functions['save_labels'](self.viewer)

    def get_instruction_text(self, config):
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

        # Create a reverse mapping of function to keys
        function_to_keys = {}
        for func, keys in config.items():
            if isinstance(keys, list):
                for key in keys:
                    if func not in function_to_keys:
                        function_to_keys[func] = []
                    function_to_keys[func].append(key)
            else:
                if func not in function_to_keys:
                    function_to_keys[func] = []
                function_to_keys[func].append(keys)

        # Function to get a string of keys for a function
        def get_key_string(func):
            keys = function_to_keys.get(func, [])
            if not keys or keys == ['']:
                return 'unassigned'
            return ' or '.join(f'<b>{key_to_string(key)}</b>' for key in keys if key)

        instruction_text = f"""
        <b>Custom Napari Keybinds:</b><br>
        - {get_key_string('toggle_labels_visibility')} to toggle label visibility<br>
        - {get_key_string('toggle_data_visibility')} to toggle data visibility<br>
        - {get_key_string('reset_plane_to_default')} to reset data plane to default position<br>
        - {get_key_string('cut_label_at_oblique_plane')} to cut label at 3D plane location<br>
        - {get_key_string('switch_to_data_layer')} to switch active layer to data layer<br>
        - {get_key_string('full_label_view')} to toggle full 3D label view<br>
        - {get_key_string('switch_to_plane_view')} to toggle 3D plane cut view layers<br>
        - {get_key_string('erase_3d_mode')} to switch to erase mode<br>
        - {get_key_string('move_mode')} to switch to pan & zoom mode<br>
        - {get_key_string('plane_erase_3d_mode')} to toggle 3D plane precision erase mode<br>
        - <b>shift + click</b> to move the 3D volume plane quickly<br>
        - <b>shift + right click + drag up or down</b> to 'fisheye' the view<br>\n
        - {get_key_string('erode_labels')} to erode labels 1 iteration<br>
        - {get_key_string('dilate_labels')} to dilate labels 1 iteration<br>
        - {get_key_string('add_padding_contextual_data')} to toggle context padding data<br>
        - {get_key_string('connected_components')} to run connected components analysis and relabel<br>
        - {get_key_string('flood_fill')} for standard flood fill<br>
        - {get_key_string('large_flood_fill')} for larger flood fill<br>
        - {get_key_string('save_labels')} to save data & labels as nrrd files<br>\n
        - {get_key_string('draw_compressed_class')} to toggle compressed region class brush<br>
        - {get_key_string('decrease_brush_size')} to decrease brush size<br>
        - {get_key_string('increase_brush_size')} to increase brush size<br>
        - {get_key_string('label_picker')} to select label layer under cursor<br>
        - {get_key_string('toggle_show_selected_label')} to toggle show selected label<br>
        - {get_key_string('interpolate_borders')} to extrapolate sparse compressed class labels<br>
        """
        return instruction_text