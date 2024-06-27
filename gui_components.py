from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton, QLabel, 
                             QVBoxLayout, QScrollArea, QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt

from helper import *
# Import necessary functions and classes from your main file
# from napari_volumetric_labelling import (erode_dilate_labels, full_label_view, switch_to_plane,
#                        add_padding_contextual_data, cut_label_at_oblique_plane,
#                        connected_components, save_labels)

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
    def __init__(self, viewer, functions_dict, update_global_erase_slice_width):
        self.viewer = viewer
        self.functions = functions_dict  # Store the functions dictionary
        self.update_global_erase_slice_width = update_global_erase_slice_width
        self.erase_slice_width = 30
        self.setup_gui()

    def setup_gui(self):
        # Create custom button widgets
        self.dilate_button = CustomButtonWidget("Dilate Labels", "u", self.dilate_labels)
        self.erode_button = CustomButtonWidget("Erode Labels", "i", self.erode_labels)
        self.full_view_button = CustomButtonWidget("Toggle Full Label View", "b", self.toggle_full_label_view)
        self.plane_cut_button = CustomButtonWidget("Toggle 3D Plane Cut View", "\\", self.toggle_3D_plane_cut_view)
        self.padding_button = CustomButtonWidget("Toggle Padding Context", "j", self.toggle_padding_context)
        self.cut_plane_button = CustomButtonWidget("Cut Label at Plane", "k", self.cut_label_at_plane_gui)
        self.components_button = CustomButtonWidget("Connected Components", "c", self.run_connected_components)
        self.save_button = CustomButtonWidget("Save Labels", "h", self.save_labels_button)
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

        instruction_label = QLabel(self.get_instruction_text())
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
    def dilate_labels(self):
        self.functions['erode_dilate_labels'](self.viewer, self.viewer.layers[self.get_label_layer_name()].data, erode=False)

    def erode_labels(self):
        self.functions['erode_dilate_labels'](self.viewer, self.viewer.layers[self.get_label_layer_name()].data, erode=True)

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

    @staticmethod
    def get_instruction_text():
        instruction_text = """
        <b>Custom Napari Keybinds:</b><br>
        - <b>/ or r</b> to toggle label visibility<br>
        - <b>. or t</b> to toggle data visibility<br>
        - <b>Left & Right arrow keys</b> move through layers<br>
        - <b>Shift + Left & Right arrow keys</b> move 20 layers<br>
        - <b>k</b> to cut label at 3D plane location<br>
        - <b>l</b> to switch active layer to data layer <br>
        - <b>b</b> to toggle full 3D label view<br>
        - <b>\\</b> to toggle 3D plane cut view layers<br>
        - <b>'</b> to switch to erase mode<br>
        - <b>;</b> to switch to pan & zoom mode<br>
        - <b>,</b> to toggle 3d plane precision erase mode<br>
        - <b>o</b> to create off-axis plane cut in 3d mode <br>
        - <b>shift + click</b> to move the 3d volume plane quickly<br>
        - <b>shift + right click + drag up or down</b> to 'fisheye' the view<br>\n
        - <b>i</b> to erode labels 1 iteration<br>
        - <b>u</b> to dilate labels 1 iteration<br>
        - <b>j</b> to toggle context padding data<br>
        - <b>c</b> to run connected components analysis and relabel<br>
        - <b>f or down arrow</b> for 20 iteration flood fill<br>
        - <b>g or up arrow</b> for 100 iteration flood fill<br>
        - <b>h</b> to save data & labels as nrrd files<br>\n
        - <b>v</b> to toggle compressed region class brush<br>
        - <b>q</b> to decrease brush size<br>
        - <b>e</b> to increase brush size<br>
        - <b>w</b> to select label layer under cursor<br>
        - <b>s</b> to toggle show selected label<br>
        - <b>a</b> to move through layers in 2d<br>
        - <b>d</b> to move through layers in 2d<br>
        - <b>x</b> to extrapolate sparse compressed class labels<br>
        """
        return instruction_text

    @staticmethod
    def get_label_layer_name():
        # Return the label name here
        return "Labels"