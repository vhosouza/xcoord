# %%

import nibabel as nib
import numpy as np

import os

# %%
data_path = 'C:\\Users\\vhosouza\\Downloads\\mri'

t1_path = os.path.join(data_path, 'EEGTMSseg_082_T1w_MPR_20200711104942_7.nii.gz')
t2_path = os.path.join(data_path, 'EEGTMSseg_082_T2w_SPC_20200711104942_9.nii.gz')

t1_data = nib.squeeze_image(nib.load(t1_path))
t1_data = nib.as_closest_canonical(t1_data)
t1_data.update_header()

t2_data = nib.squeeze_image(nib.load(t2_path))
t2_data = nib.as_closest_canonical(t2_data)
t2_data.update_header()

# %%
print(np.array_equal(t1_data.get_qform(), t1_data.get_sform()))