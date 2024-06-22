Install instructions with conda:

```
conda create --name napari_vesuvius
conda activate napari_vesuvius
pip install -r requirements.txt
```

Attach the napari_vesuvius environment as the kernel to the napari_volumetric_labelling.ipynb jupyter notebook
Change path variables to point to your raw data cube and semantic mask cube (commented code for zarrs and individual nrrds already exists in the notebook, though anything that loads in a numpy array can work and be assigned to the data and label variables)

Watch the tutorial video on how to use

Optionally search for 'keybind' in the notebook to change the keys certain custom functions are mapped to.

Please contact me with a github issue or @james darby on the Vesuvisu challenge discord server if you have a feature request or pain point description that would help volumetrically label data faster or more accurately. 