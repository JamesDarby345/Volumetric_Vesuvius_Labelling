## Purpose of the repo:
To provide custom tooling and extensions to the Napari 3d viewer that will help create more and better manually annotated volumetric masks of the Vesuvius Scrolls Data. These labels can be used to train ML Networks to seperate sheets automatically, and make inroads to automatic segmentation, inclusing in dense or compressed regions

## Features:
-2D & 3D volumetric label editing<br>
-Responsive 3D Erasing<br>
-Oblique, 3D off-axis cuts, especially useful in dense and confusing regions (most of scroll 2)<br>
-Limited range flood fill to quickly find connections in 2d, and to serve as a landmark in 3d.<br>
-One button extended cube context <br>
-Label erosion & dilation<br>
-Connected Components seperation<br>
-Compressed/Mush class<br>
-Editable Hotkeys specific to the target task for switching between modes, tools and views to increase speed & fluidity of work<br>

Watch the tutorial video on how to use available here: https://dl.ash2txt.org/community-uploads/james/Napari_Volumetric_Cube_Labelling_Tutorial.mp4 <br>

![Screenshot 2024-06-24 at 7 13 04 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/23372150-e319-414d-b6bf-63a2c5b85ee6)

## Install instructions with conda:

```
conda create --name napari_vesuvius
conda activate napari_vesuvius
pip install -r requirements.txt
```

Change path variables to point to your raw data cube and semantic mask cube, or move them to a data/nrrd_cubes folder beside the scripts which is the default expected path. (Commented code for zarrs and individual nrrds already exists in the notebook, though anything that loads in a numpy array can work and be assigned to the data and label variables)

Attach the napari_vesuvius environment as the kernel to the napari_volumetric_labelling.ipynb jupyter notebook or activate the conda environment and run the napari_volumetric_labelling.py file.

Search for Change to find path location variables to change to point to raw and mask data files (zarr and nrrd raw files supported, nrrd and jordi's blosc2 gross volumetric labels supported)

Optionally search for 'keybind' in the notebook to change the keys certain custom functions are mapped to. Note that keys like x,y,z,[,],-,= have default Napari mappings and may not work in all modes.

## Custom Napari Keybinds:<br>
/ or r to toggle label visibility<br>
. or t to toggle data visibility<br>
Left & Right arrow keys scrub through layers in 2D & 3D planes<br>
k to cut label at 3D plane location, toggles displayed side of cut<br>
l to switch active layer to data layer (useful to move 3d plane with shift click)<br>
b to toggle between 2D & 3D views with the full 3D label visible<br>
\ to toggle between 2D & 3D views and setup plane cut view layers<br>
' to switch to erase mode<br>
; to switch to pan & zoom mode<br>
, to toggle 3d plane precision erase mode<br>
o to create off-axis plane cut in 3d mode (can also hold down to adjust)<br>
shift + click to move the 3d volume plane quickly<br>
shift + right click + drag up or down to 'fisheye' the view, useful to zoom into structures, note that erase etc doesnt work in this mode :(<br>

i to erode labels 1 iteration<br>
u to dilate labels 1 iteration<br>
j to toggle context padding data<br>
c to run connected components analysis and relabel<br>
f or down arrow for 20 iteration flood fill<br>
g or up arrow for 100 iteration flood fill<br>

h to save data & labels as nrrd files<br>

v to toggle compressed region class brush<br>
q to decrease brush size<br>
e to increase brush size<br>
w to select label layer that was last clicked in move mode, alternatively use color picker (4)<br>
s to toggle show selected label<br>
a to move through layers in 2d<br>
d to move through layers in 2d<br>
x to extrapolate sparse compressed class labels<br>

Note that moving the 3d plane with the arrows keys with the label visible (without hiding with /) can be laggy if moving more than one or two slices, use shift + click with the data layer selected (hotkey of l to select the data layer) or hide the label with / for better performance.

Please contact me with a github issue or @james darby on the Vesuvius challenge discord server if you have a feature request or pain point description that would help volumetrically label data faster or more accurately. 

![Screenshot 2024-06-24 at 7 13 12 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/d50617ff-e159-4710-ada5-feb51852a334)
![Screenshot 2024-06-24 at 7 13 36 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/124f9d0b-090c-4e92-b009-ab8a2d083428)
