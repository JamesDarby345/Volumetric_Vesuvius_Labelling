## Purpose of the repo:
To provide custom tooling and extensions to the Napari 3d viewer that will help create manual and semi-automatic annotated volumetric masks of the Vesuvius Scrolls Data. These labels can be used to train ML Networks to separate sheets automatically, and make inroads to automatic segmentation, including in dense or compressed regions. The repo has also been extended to explore, and edit ink detection volumes in 3d. This functionality allows for inspection of ink signal in unsegmented regions and 3d ink label refinement that allows for Human in the Loop ML cycles to hopefully find ink signal in the other scrolls.

## Features:
-2D & 3D volumetric label editing<br>
-Sheet midline push and nearest voxel assignment for semi-automated labelling<br>
-Responsive 3D Erasing<br>
-Segmentation mesh label coloring for semi-automated voumetric labelling<br>
-3D Chunk View to limit view to relevant context<br>
-Oblique, 3D off-axis cuts, especially useful in dense and confusing regions<br>
-Limited range flood fill to quickly find connections in 2d, and to serve as a landmark in 3d<br>
-Single button extended cube context<br>
-Label erosion & dilation<br>
-Connected Components separation and preview<br>
-Editable Hotkeys specific to the target task for switching between modes, tools and views to increase speed & fluidity of work<br>

Watch the tutorial video on how to use it: https://dl.ash2txt.org/community-uploads/james/Updated_Napari_Volumetric_Labeling_Tutorial.mp4 <br>

## Install with conda & setup instructions:
Run these commands in the repo folder after installing miniconda: https://docs.anaconda.com/miniconda/
```
conda env create -f environment.yml
```
```
conda activate napari_vesuvius
```

Run the software with (see setup instructions first):
```
python napari_volumetric_labelling.py
```

### Nrrd cube setup
Create a /data/nrrd_cubes/s1 folder in the repository to put the nrrd cubes from the download server into. The s1 folder is the scroll_name value in the config file, it can be changed to keep separate scroll cubes apart. The code is expecting the folder structure of zzzzz_yyyyy_xxxxx/zzzzz_yyyyy_xxxxx_volume.nrrd and zzzzz_yyyyy_xxxxx_mask.nrrd, just like how it is setup on the download server. Link to location of s1 cubes on the Vesuvius download server: https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/cubes/

### Local config setup
Copy the napari_config.yaml file and rename it local_napari_config.yaml
Edit the z, y, x coordinates to match the cube you want.
You can also edit the hotkey buttons in the config file.

### Zarr volume setup (Recommended)
Additionally, if you have the scroll zarr volumes downloaded, you can specify the path to it with the raw_data_zarr_path. Note that to use provided masks, use the folder structure for the nrrd cubes. This allows the additional context function to always have the raw data context. Link to scroll 1 zarr volume (~980 GB): https://dl.ash2txt.org/community-uploads/james/Scroll1/ Link to scroll 2 zarr volume (2.2TB): https://dl.ash2txt.org/community-uploads/james/Scroll2/

### Semi-Automated volumetric labelling setup
Create a data/manual_sheet_segmentation/s1 folder, and put the segment id folders containing the voxelized GP segmentation meshes (the .b2nd and origin.txt files) in it. Link to voxelized GP segmentation meshes: https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/GP-banner/. <br> Rclone download command (change local path): 
```
rclone copy --http-url https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/GP-banner/ :http: path/to/local/folder/Volumetric_Vesuvius_Labelling/data/manual_sheet_segmentation/s1 --multi-thread-streams 8 --transfers 8 --progress
```

### Ink Detection setup
If you want to use the software for 3d ink detection volume inspection and label refinement, download or create a 3d ink detection zarr. <br>
Link to bruniss' 3d ink detection zarrs: https://dl.ash2txt.org/community-uploads/bruniss/3d%20Ink%20/ <br>
Link to emel ryan's 3d ink detection zarrs: https://dl.ash2txt.org/community-uploads/ryan/ <br>
Specify the path to the ink detection zarr in the config with the ink_pred_zarr_path and change the main label layer name to 'ink'. If you are using the matching raw data zarr file, the axis should automatically align, if you are using nrrd cubes, you will have to specify the correct axis order with the raw_data_axis_order: 'zyx' and ink_pred_label_order: 'zyx' variables so the ink label and raw data align. Additionally if you are using the zarr, you can set the chunk_size to be larger so you can view more of the volume. If the ink detection volume uses a value range, set the ink_threshold value to provide a cutoff. Additionally you may want to set create_papyrus_mask_if_not_provided to False so papyrus labels arent created, which slows down load times. Scroll 1 8um scan letters are approximately 400 voxels tall. 

### How to use for volumetric labelling
Workflow for manual & semi automated volumetric cube labelling : https://docs.google.com/document/d/1NVG1L2rLySrzstuXc2LvHFkdnJ0PZx1nwPMcYrs1fHM/edit?usp=sharing

### How to use for 3d ink detection label inspection and refinement
Workflow to view the 3d ink detection labels and manually refine them: https://docs.google.com/document/d/1i-v9Qv2bNEm7CP5vXZHKkSAPX_S1nIStXoC2frAkg3E/edit?usp=sharing

Please contact me with a github issue or \@james darby on the Vesuvius challenge discord server if you have a feature request or pain point description that would help volumetrically label data faster or more accurately.

![Screenshot 2024-06-30 at 8 23 48 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/10d8cd2d-50d9-4c08-b112-9579923354a6)
![Screenshot 2024-06-30 at 8 24 40 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/b2b552c1-70d9-4ae7-baf7-19f66a5852e6)
![Screenshot 2024-06-30 at 8 26 57 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/759d7816-b6f3-4967-bd5b-c8a0cd275b77)
![Screenshot 2024-06-30 at 8 29 00 PM](https://github.com/JamesDarby345/Volumetric_Vesuvius_Labelling/assets/49734270/471096b0-bdea-4d72-8616-8a1af9533977)
