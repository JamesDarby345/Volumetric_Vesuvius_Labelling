import SimpleITK as sitk
import nrrd
path1 = '/Users/jamesdarby/Downloads/08400_04304_02768_mask.nrrd'
path2 = '/Users/jamesdarby/Documents/VesuviusScroll/GP/Volumetric_Vesuvius_Labelling/output/volumetric_labels_s1/08400_04304_02768/08400_04304_02768_zyx_256_chunk_s1_vol_label.nrrd'
path = path2
image = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(image)
print("sitk worked")

nrrd.read(path)
print("pynrrd worked")

