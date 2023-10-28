#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np

# from skimage import exposure
# from skimage.util import img_as_int

import image_funcs as imf


def image_normalize(image, min_=0.0, max_=1.0, output_dtype=np.int16):
    output = np.empty(shape=image.shape, dtype=output_dtype)
    imin, imax = image.min(), image.max()
    output[:] = (image - imin) * ((max_ - min_) / (imax - imin)) + min_
    return output


# %%
data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\pixel_rescale'
filenames = ['joonas', 'victoria', 'agnese']
filepaths = [os.path.join(data_dir, n + '.nii') for n in filenames]

imagedata = [imf.load_image(n)[0] for n in filepaths]

matrix_range = [imf.img2memmap(n) for n in imagedata]
# matrix, scalar_range = imf.img2memmap(group)

# matrix_int = [img_as_int(n[0]) for n in matrix_range]

# image = exposure.rescale_intensity(matrix_range[1][0], in_range='float')
# image = exposure.rescale_intensity(matrix_range[1][0], in_range=(0, 1))

# %%
# uint16 maximum number (2**16/2-1)
scalar_range_init = np.amin(matrix_range[1][0]), np.amax(matrix_range[1][0])

image_normalized = image_normalize(matrix_range[1][0], min_=0, max_=10000, output_dtype=np.int16)
image_normalized2 = matrix_range[1][0].astype('uint16')

new_min, new_max = 0, 10000
current_min, current_max = matrix_range[1][0].min(), matrix_range[1][0].max()
image_unscalled = matrix_range[1][0].copy()
image_rescaled = (new_max - new_min) * (image_unscalled - current_min) / (current_max - current_min) + new_min
image_rescaled = image_rescaled.astype('int16')
# image_skimage = exposure.rescale_intensity(image_rescaled, out_range='int16')

# %
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))
imgplot = axs[0, 0].imshow(matrix_range[1][0][:, :, 128], cmap="gray")
fig.colorbar(imgplot, ax=axs[0, 0], orientation='horizontal')
# imgplot = plt.imshow(b[:, :, 128], cmap="gray")
imgplot = axs[0, 1].imshow(image_rescaled[:, :, 128], cmap="gray")
fig.colorbar(imgplot, ax=axs[0, 1], orientation='horizontal')
imgplot = axs[1, 0].imshow(image_normalized[:, :, 128], cmap="gray")
fig.colorbar(imgplot, ax=axs[1, 0], orientation='horizontal')
imgplot = axs[1, 1].imshow(image_normalized2[:, :, 128], cmap="gray")
fig.colorbar(imgplot, ax=axs[1, 1], orientation='horizontal')
# plt.hist(matrix_range[1][0].ravel(), bins=256, fc='k', ec='k')
# plt.colorbar()
fig.tight_layout()
plt.show()

# %%
for n in matrix_range:
    print("Scalar range: ", n[1])
    print("Type: ", n[0].dtype)

