from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel, 
                             QVBoxLayout, QScrollArea, QSizePolicy, QSpinBox)
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
    def __init__(self, viewer, functions_dict, update_global_erase_slice_width, config):
        self.viewer = viewer
        self.functions = functions_dict  # Store the functions dictionary
        self.update_global_erase_slice_width = update_global_erase_slice_width
        self.erase_slice_width = 30
        self.config = config
        self.setup_gui()

    def get_key_string(self, func):
        keys = self.config.get(func, [])
        if isinstance(keys, list):
            return ' or '.join(keys)
        return str(keys)

    def setup_gui(self):
        # Create custom button widgets
        self.dilate_button = CustomButtonWidget("Dilate Labels", self.get_key_string('dilate_labels'), self.dilate_labels_gui)
        self.erode_button = CustomButtonWidget("Erode Labels", self.get_key_string('erode_labels'), self.erode_labels_gui)
        self.full_view_button = CustomButtonWidget("Toggle Full Label View", self.get_key_string('full_label_view'), self.toggle_full_label_view)
        self.plane_cut_button = CustomButtonWidget("Toggle 3D Plane Cut View", self.get_key_string('switch_to_plane'), self.toggle_3D_plane_cut_view)
        self.padding_button = CustomButtonWidget("Toggle Padding Context", self.get_key_string('add_padding_contextual_data'), self.toggle_padding_context)
        self.cut_plane_button = CustomButtonWidget("Cut Label at Plane", self.get_key_string('cut_label_at_oblique_plane'), self.cut_label_at_plane_gui)
        self.components_button = CustomButtonWidget("Connected Components", self.get_key_string('connected_components'), self.run_connected_components)
        self.save_button = CustomButtonWidget("Save Labels", self.get_key_string('save_labels'), self.save_labels_button)
    
        color_picker_widget = ColorPickerWidget(self.viewer)

        # Create erase width input
        erase_slice_width_layout = QHBoxLayout()
        erase_slice_width_label = QLabel("Erase slice Width:")
        self.erase_slice_width_spinbox = QSpinBox()
        self.erase_slice_width_spinbox.setMinimum(1)
        self.erase_slice_width_spinbox.setMaximum(100)
        self.erase_slice_width_spinbox.setValue(self.erase_slice_width)
        self.erase_slice_width_spinbox.valueChanged.connect(self.update_erase_slice_width)
        erase_slice_width_layout.addWidget(erase_slice_width_label)
        erase_slice_width_layout.addWidget(self.erase_slice_width_spinbox)

        # Create a container widget for buttons
        button_container_widget = QWidget()
        button_container_layout = QVBoxLayout()
        button_container_widget.setLayout(button_container_layout)

        # Add buttons to the container
        for button in [self.dilate_button, self.erode_button, self.full_view_button, self.plane_cut_button, 
                       self.cut_plane_button, self.padding_button, self.components_button, self.save_button]:
            button_container_layout.addWidget(button)

        button_container_layout.addLayout(erase_slice_width_layout)
        button_container_layout.addWidget(color_picker_widget)

        # Create a container widget for the instruction text with scrollable area
        text_container_widget = QWidget()
        text_container_layout = QVBoxLayout()
        text_container_widget.setLayout(text_container_layout)

        instruction_label = QLabel(self.get_instruction_text(self.config))
        instruction_label.setWordWrap(True)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        text_container_layout.addWidget(instruction_label)

        # Add the text container to a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(text_container_widget)

        # Create a main container widget to hold both button and text containers
        main_container_widget = QWidget()
        main_container_layout = QVBoxLayout()
        main_container_widget.setLayout(main_container_layout)

        # Adjust size policies and layout properties
        main_container_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        main_container_layout.setSpacing(5)

        # Add button container to the main container
        main_container_layout.addWidget(button_container_widget)

        # Add the main container to the viewer
        self.viewer.window.add_dock_widget(main_container_widget, area='right')

        # Add the scroll area to the viewer as a separate dock widget
        self.viewer.window.add_dock_widget(scroll_area, area='right')

    def update_erase_slice_width(self, value):
        self.erase_slice_width = value
        self.update_global_erase_slice_width(value)
        print(f"Erase width updated to: {self.erase_slice_width}")

    def setup_napari_defaults(self):
        self.viewer.axes.visible = True
        labels_layer = self.viewer.layers[self.get_label_layer_name()]
        labels_layer.n_edit_dimensions = 3
        labels_layer.opacity = 1
        labels_layer.contour = 1
        self.viewer.theme = 'light'
        self.viewer.window._qt_viewer.canvas.bgcolor = (0.68, 0.85, 0.90, 1.0)
        labels_layer.colormap = get_direct_label_colormap()
        labels_layer.shape = 'square'
        self.viewer.layers.selection.active = labels_layer

    # Define button callback methods
    def dilate_labels_gui(self):
        self.functions['dilate_labels'](self.viewer)

    def erode_labels_gui(self):
        self.functions['erode_labels'](self.viewer)

    def toggle_full_label_view(self):
        self.functions['full_label_view'](self.viewer)

    def toggle_3D_plane_cut_view(self):
        self.functions['switch_to_plane'](self.viewer)

    def toggle_padding_context(self):
        self.functions['add_padding_contextual_data'](self.viewer)

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
                'Shift-Left': 'Shift + Left Arrow',
                'Shift-Right': 'Shift + Right Arrow',
                'Shift-Up': 'Shift + Up Arrow',
                'Shift-Down': 'Shift + Down Arrow',
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
            return ' or '.join(f'<b>{key_to_string(key)}</b>' for key in keys if key)

        instruction_text = f"""
        <b>Custom Napari Keybinds:</b><br>
        - {get_key_string('toggle_labels_visibility')} to toggle label visibility<br>
        - {get_key_string('toggle_data_visibility')} to toggle data visibility<br>
        - {get_key_string('shift_plane_left')} & {get_key_string('shift_plane_right')} to move through layers<br>
        - {get_key_string('shift_plane_left_fast')} & {get_key_string('shift_plane_right_fast')} to move through layers faster<br>
        - {get_key_string('cut_label_at_oblique_plane')} to cut label at 3D plane location<br>
        - {get_key_string('switch_to_data_layer')} to switch active layer to data layer<br>
        - {get_key_string('full_label_view')} to toggle full 3D label view<br>
        - {get_key_string('switch_to_plane')} to toggle 3D plane cut view layers<br>
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
        - {get_key_string('shift_dim_left')} to shift dimension left in 2D<br>
        - {get_key_string('shift_dim_right')} to shift dimension right in 2D<br>
        - {get_key_string('interpolate_borders')} to extrapolate sparse compressed class labels<br>
        """
        return instruction_text

    @staticmethod
    def get_label_layer_name():
        # Return the label name here
        return "Labels"