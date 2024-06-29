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
See the napari_config.yaml file for the key binds

Note that you can create a local_napari_config.yaml file which will be read instead and is in thegitignore if you want to avoid conflicts if the config file is changed when pulling updates

Note that moving the 3d plane with the arrows keys with the label visible (without hiding with /) can be laggy if moving more than one or two slices, use shift + click with the data layer selected (hotkey of l to select the data layer) or hide the label with / for better performance.

Please contact me with a github issue or @james darby on the Vesuvius challenge discord server if you have a feature request or pain point description that would help volumetrically label data faster or more accurately. 

![Screenshot 2024-06-24 at 7 13 12 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/d50617ff-e159-4710-ada5-feb51852a334)
![Screenshot 2024-06-24 at 7 13 36 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/124f9d0b-090c-4e92-b009-ab8a2d083428)
