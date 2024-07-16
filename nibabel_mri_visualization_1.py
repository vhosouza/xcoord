#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import sys

import external.transformations as tf

import image_funcs as imf

import nibabel as nb


# %%
if sys.platform == "win32":
    onedrive_path = os.environ.get('OneDrive')
elif (sys.platform == "darwin") or (sys.platform == "linux"):
    onedrive_path = os.path.expanduser('~/OneDrive - Aalto University')
else:
    onedrive_path = False
    print("Unsupported platform")

data_dir = os.path.join(onedrive_path, 'projects', 'nexstim', 'data', 'mri')
filenames = 'GJ_2008_anonym_t1_mpr_ns_sag_1_1_1_mm_20081021180940_2'
filename_efield = 'nexstim_efield_map_1000'

# data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\pixel_rescale'
# filenames = ['joonas', 'victoria', 'agnese']
filepaths = os.path.join(data_dir, f'{filenames}.nii.gz')
filepath_efield = os.path.join(data_dir, f'{filename_efield}.csv')

imagedata = nb.squeeze_image(nb.load(filepaths))
imagedata = nb.as_closest_canonical(imagedata)
imagedata.update_header()

affine = imagedata.affine
scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
affine_noscale = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

pix_dim = [imagedata.header.get_zooms()[n] for n in [2, 1, 0]]
img_shape = [imagedata.header.get_data_shape()[n] for n in [2, 1, 0]]

# %%

# Get the indices
indices = np.indices(img_shape)
# Transpose and reshape to get triplets
image_coordinates = np.transpose(indices).reshape(-1, 3)
image_coordinates = np.hstack([image_coordinates + 1, np.ones((image_coordinates.shape[0], 1))])
image_coordinates_w = imagedata.affine @ image_coordinates.T
image_coordinates_w = image_coordinates_w.T
orig = np.array([[0., 0., 0., 1.]])
# replace 'file_path.csv' with your file path
efield = pd.read_csv(filepath_efield, delimiter=';')
efield_array = efield.values[:, [0, 1, 2]]
# efield_array = efield.values[:, [2, 1, 0]]

# image_array = image_array[::-1, :, :]
print("max: ", np.max(image_coordinates_w, axis=0))
print("min: ", np.min(image_coordinates_w, axis=0))

# %%
efield_array_w = efield.to_numpy(copy=True)
efield_array_w[:, -1] = 1.

# Assuming z is your 2D array
# Creating x, y coordinates for the 2D array
n_slice = 400

z = image_array[n_slice, :, :]
x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
img_coordinates = np.array([x.flatten(), 0.5*y.flatten()]).T
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(img_coordinates[:, 0], img_coordinates[:, 1])

z_mri = mri_array[n_slice, :, :]
x_mri, y_mri = np.meshgrid(np.arange(z_mri.shape[1]), np.arange(z_mri.shape[0]))
mri_coordinates = np.array([x_mri.flatten(), 0.5*y_mri.flatten()]).T
triang_mri = tri.Triangulation(mri_coordinates[:, 0], mri_coordinates[:, 1])

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False)

# %
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))

tcf = axs[0].tricontourf(triang, z.flatten(), cmap='gray', levels=255)
tcf = axs[1].tricontourf(triang_mri, z_mri.flatten(), cmap='gray', levels=255)

for ax in axs:
    ax.set_aspect('equal')

# Add a color bar which maps values to colors.
# fig.colorbar(tcf, shrink=0.5, aspect=5)

plt.show()



# %%
# Assuming z is your 2D array
# Creating x, y coordinates for the 2D array
n_slice = 400

z = image_array[n_slice, :, :]
x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
img_coordinates = np.array([x.flatten(), 0.5*y.flatten()]).T
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(img_coordinates[:, 0], img_coordinates[:, 1])

z_mri = mri_array[n_slice, :, :]
x_mri, y_mri = np.meshgrid(np.arange(z_mri.shape[1]), np.arange(z_mri.shape[0]))
mri_coordinates = np.array([x_mri.flatten(), 0.5*y_mri.flatten()]).T
triang_mri = tri.Triangulation(mri_coordinates[:, 0], mri_coordinates[:, 1])

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False)

# %
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))

tcf = axs[0].tricontourf(triang, z.flatten(), cmap='gray', levels=255)
tcf = axs[1].tricontourf(triang_mri, z_mri.flatten(), cmap='gray', levels=255)

for ax in axs:
    ax.set_aspect('equal')

# Add a color bar which maps values to colors.
# fig.colorbar(tcf, shrink=0.5, aspect=5)

plt.show()


# %%
# Create a figure
fig = plt.figure()
# Create a 3d Axes
ax = fig.add_subplot(111, projection='3d')
# cmap = plt.get_cmap('cividis')
# colororder = [cmap(i) for i in efield.values[:, 3]]
# Add surface plot
ax.plot_trisurf(efield.values[:, 0], efield.values[:, 1], efield.values[:, 2], antialiased=True)
ax.set_aspect('equal')
# Show the figure
plt.show()


# %%
fig, ax = plt.subplots()
ax.set_aspect('equal')
tcf = ax.tricontourf(triang, z.flatten(), cmap='gray', levels=255)
plt.scatter([350], [300], marker='o', color='r')

# Add a color bar which maps values to colors.
fig.colorbar(tcf, shrink=0.5, aspect=5)

plt.show()


# %%
# %
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(3, 3))
imgplot = axs.imshow(image_array[:, :, 128], cmap="gray")
fig.colorbar(imgplot, ax=axs, orientation='horizontal')
# plt.hist(matrix_range[1][0].ravel(), bins=256, fc='k', ec='k')
# plt.colorbar()
fig.tight_layout()
plt.show()


# %%
# compare if reordering axes in array like in invesalius is working fine

affine = np.linalg.inv(affine)  # by default invesalius affine is the inverted affine from the header
# this is how the image is loaded in the fmri panel
mri_data = nb.squeeze_image(nb.load(filepaths))
mri_data = nb.as_closest_canonical(mri_data)
mri_data.update_header()
mri_array = mri_data.get_fdata().T[:, ::-1]

# the original mri array with reordering
mri_array_0 = mri_data.get_fdata()

print("changed and original: ", mri_array.shape, mri_array_0.shape)

# Assuming z is your 2D array
# Creating x, y coordinates for the 2D array
n_slice = 400

z = image_array[n_slice, :, :]
x, y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
img_coordinates = np.array([x.flatten(), 0.5*y.flatten()]).T
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(img_coordinates[:, 0], img_coordinates[:, 1])

z_mri = mri_array[n_slice, :, :]
x_mri, y_mri = np.meshgrid(np.arange(z_mri.shape[1]), np.arange(z_mri.shape[0]))
mri_coordinates = np.array([x_mri.flatten(), 0.5*y_mri.flatten()]).T
triang_mri = tri.Triangulation(mri_coordinates[:, 0], mri_coordinates[:, 1])

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False)

# %
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))

tcf = axs[0].tricontourf(triang, z.flatten(), cmap='gray', levels=255)
tcf = axs[1].tricontourf(triang_mri, z_mri.flatten(), cmap='gray', levels=255)

for ax in axs:
    ax.set_aspect('equal')

# Add a color bar which maps values to colors.
# fig.colorbar(tcf, shrink=0.5, aspect=5)

plt.show()