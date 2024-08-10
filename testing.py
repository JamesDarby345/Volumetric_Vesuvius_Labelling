import SimpleITK as sitk
path1 = '/Users/jamesdarby/Documents/VesuviusScroll/GP/Volumetric_Vesuvius_Labelling/output/volumetric_labels_s1/09936_03792_05072/09936_03792_05072_zyx_256_chunk_s1_vol_label.nrrd'
path2 = '/Users/jamesdarby/Documents/VesuviusScroll/GP/Volumetric_Vesuvius_Labelling/output/volumetric_labels_s1/10192_03792_05072_debug/10192_03792_05072_zyx_256_chunk_s1_vol_label.nrrd'
image = sitk.ReadImage(path2)
data = sitk.GetArrayFromImage(image)


