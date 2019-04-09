import numpy as np
import nibabel as nib

filepath_image = "D:\\Programas\\Google Drive\\Lab\\Doutorado\\projetos\\DTI_coord_transf\\nexstim_coordinates\\OriginalImage\\5_mprage_mgh-variant.nii"

#filepath_data = "D:\\Programas\\Google Drive\\Lab\\Doutorado\\projetos\\DTI_coord_transf\\nexstim_coordinates\\nexstim_coords.mks"
filepath_data = "D:\\Programas\\Google Drive\\Lab\\Doutorado\\projetos\\DTI_coord_transf\\nexstim_coordinates\\nexstim_coords.mks"

imagedata = nib.squeeze_image(nib.load(filepath_image))
imagedata = nib.as_closest_canonical(imagedata)
imagedata.update_header()
hdr = imagedata.header

data = np.loadtxt(filepath_data)
data_flip = data[:,0:3]
i = np.argsort([0,2,1])
data_flip = data[:,i]
data_flip[:,1] = hdr.get_data_shape()[1] - data[:,1]
data_flip[:,0] = hdr.get_data_shape()[0] - data[:,0]

NBS2INV_markers = np.hstack((data_flip, data[:,3:]))

np.savetxt("D:\\Programas\\Google Drive\\Lab\\Doutorado\\projetos\\DTI_coord_transf\\nexstim_coordinates\\NBS2INV_markers.mks",NBS2INV_markers)