## Purpose of the repo:
To provide custom tooling and extensions to the Napari 3d viewer that will help create more and better manually annotated volumetric masks of the Vesuvius Scrolls Data. These labels can be used to train ML Networks to seperate sheets automatically, and make inroads to automatic segmentation, inclusing in dense or compressed regions.

## Features:
-2D & 3D volumetric label editing<br>
-Responsive 3D Erasing<br>
-3D Chunk View to limit view to relevant context<br>
-Oblique, 3D off-axis cuts, especially useful in dense and confusing regions (most of scroll 2)<br>
-Limited range flood fill to quickly find connections in 2d, and to serve as a landmark in 3d.<br>
-One button extended cube context<br>
-Label erosion & dilation<br>
-Connected Components seperation and preview<br>
-Editable Hotkeys specific to the target task for switching between modes, tools and views to increase speed & fluidity of work<br>

Watch the tutorial video on how to use available here: https://dl.ash2txt.org/community-uploads/james/Updated_Napari_Volumetric_Labeling_Tutorial.mp4 <br>

## Install with conda & setup instructions:

```
conda create --name napari_vesuvius
conda activate napari_vesuvius
pip install -r requirements.txt
```
Create a /data/nrrd_cubes/s1 folder in the repository to put the nrrd cubes from the download server into. The s1 is the scroll_name value in the config, it can be changed to keep seperate scroll cubes apart. Link to location of s1 cubes on the Vesuvius download server: https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/cubes_renamed/

Copy the napari_config.yaml file and rename it local_napari_config.yaml
Edit the z, y, x coordinates to match the ones of the cube you want to edit.
You can also edit the hotkey buttons in the config file. 

Please contact me with a github issue or @james darby on the Vesuvius challenge discord server if you have a feature request or pain point description that would help volumetrically label data faster or more accurately. 

![Screenshot 2024-06-30 at 8 23 48 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/10d8cd2d-50d9-4c08-b112-9579923354a6)
![Screenshot 2024-06-30 at 8 24 40 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/b2b552c1-70d9-4ae7-baf7-19f66a5852e6)
![Screenshot 2024-06-30 at 8 26 57 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/759d7816-b6f3-4967-bd5b-c8a0cd275b77)
![Screenshot 2024-06-30 at 8 29 00 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/471096b0-bdea-4d72-8616-8a1af9533977)
