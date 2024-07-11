import zarr
import napari
import dask.array as da
import numpy as np
from helper import bright_spot_mask_dask

viewer = napari.Viewer()
chunk_size = 512
x = 2000
y = 2000
z = 2000

# Ink detection data
zarr_ink_det_path = "/Users/jamesdarby/Desktop/bruniss_3d_ink_preds/scroll3_predictions.zarr/0.zarr"
zarr_ink_det_path = "/Volumes/16TB_RAID_0/3d_ink_zarrs/3d_predictions_scroll3.zarr"
z1 = zarr.open(zarr_ink_det_path, mode="r")
print("Ink detection data info:")
print(z1.info)
d = da.from_zarr(z1)
d = d[x:x+chunk_size, y:y+chunk_size, z:z+chunk_size]

# Raw data
zarr_raw_data_path = "/Volumes/16TB_slow_RAID_0/Scroll3/Scroll3_8um.zarr"
zarr2 = zarr.open(zarr_raw_data_path, mode="r")
raw_zarr = zarr2[0]
print("Raw data info:")
print(raw_zarr.info)

d2 = da.from_zarr(raw_zarr)
d2 = d2[z:z+chunk_size, x:x+chunk_size, y:y+chunk_size]

bright_spot_mask_arr = bright_spot_mask_dask(d2)
d2[bright_spot_mask_arr] = 30000

d = d.transpose((2, 0, 1))
print("Shape of ink detection data after transposing:", d.shape)

viewer.add_image(
    d,
    contrast_limits=[0, 255],
    scale=[1, 1, 1],  # Adjust if your voxels aren't isotropic
    name='Ink Detection Data'
)
thresholded_data = da.where(d < 150, 0, 1)
viewer.add_labels(
    thresholded_data,
    scale=[1, 1, 1],  # Adjust if your voxels aren't isotropic
    name='Ink Detection Data',
    blending='translucent'
)

viewer.add_image(
    d2,
    contrast_limits=[0, 65535],
    scale=[1, 1, 1],  # Adjust this scale factor if needed
    name='Raw Data',
    depiction='plane',
    blending='translucent'
)
# widget = QtRenderPlaneManipulatorWidget(viewer)
# dock_widget = QDockWidget("Plane Manipulator")
# dock_widget.setObjectName("plane_manipulator_dock")  # Set a unique object name
# dock_widget.setWidget(widget)
# viewer.window.add_dock_widget(dock_widget, area="right")
viewer.dims.ndisplay = 3   
napari.run()