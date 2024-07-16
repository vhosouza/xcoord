#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import sys

from scipy.interpolate import griddata

import external.transformations as tf

import image_funcs as imf

import nibabel as nb


# %
if sys.platform == "win32":
    onedrive_path = os.environ.get('OneDrive')
elif (sys.platform == "darwin") or (sys.platform == "linux"):
    onedrive_path = os.path.expanduser('~/OneDrive - Aalto University')
else:
    onedrive_path = False
    print("Unsupported platform")

data_dir = os.path.join(onedrive_path, 'projects', 'nexstim', 'data', 'mri')
filenames = 'GJ_2008_anonym_t1_mpr_ns_sag_1_1_1_mm_20081021180940_2'
filename_efield_extended = 'nexstim_efield_map_1000_extended'
filename_efield = 'nexstim_efield_map_1000_rtp'

# data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\pixel_rescale'
# filenames = ['joonas', 'victoria', 'agnese']
filepaths = os.path.join(data_dir, f'{filenames}.nii.gz')
#filepath_efield_extended = os.path.join(data_dir, f'{filename_efield_extended}.csv')
filepath_efield = os.path.join(data_dir, f'{filename_efield}.csv')

#efield_extended = pd.read_csv(filepath_efield_extended, delimiter=';')
#efield_xyz_extended = efield_extended.to_numpy(copy=True)

efield = pd.read_csv(filepath_efield, delimiter=';')
efield_rtp = efield.to_numpy(copy=True)
#efield_xyz = efield.to_numpy(copy=True)
#efield_xyz[:, -1] = 1.

imagedata = nb.squeeze_image(nb.load(filepaths))
imagedata = nb.as_closest_canonical(imagedata)
imagedata.update_header()
image_array = imagedata.get_fdata()

affine = imagedata.affine
scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
affine_noscale = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

pix_dim = imagedata.header.get_zooms()
img_shape = imagedata.header.get_data_shape()

# %
start, end, step = 0.068, 0.073, 0.0005
new_radii = np.arange(start, end, step)
efield_rtp_extended = []
for r in new_radii:
    efield_rtp_aux = efield_rtp.copy()
    efield_rtp_aux[:, 0] = r
    efield_rtp_extended.append(efield_rtp_aux)

efield_rtp_extended = np.vstack(efield_rtp_extended)
efield_xyz = efield_rtp_extended.copy()
efield_xyz[:, :3] = 1e3*imf.polar_to_cartesian(efield_rtp_extended[:, :3])

# %%
# Get the indices
indices = np.indices(img_shape)
# Transpose and reshape to get triplets
image_coordinates = np.transpose(indices).reshape(-1, 3)
image_coordinates = image_coordinates.astype(np.float64) + 1.
image_coordinates_homo = np.hstack((image_coordinates, np.ones((image_coordinates.shape[0], 1))))
image_coordinates_homo_slice = image_coordinates_homo[:5, :].copy()
#print(image_coordinates_homo[:3, :])

# compute 3D array MRI indices coordinates in worlds/scanner/xyz space
xyz = np.transpose(affine @ image_coordinates_homo.T)[:, :3]
#del image_coordinates_homo, xyz
xyz_min, xyz_max = np.min(xyz[:, :3], axis=0), np.max(xyz[:, :3], axis=0)
xyz_min_max = [n for n in zip(xyz_min, xyz_max)]

xyz_rtp = xyz.copy()
xyz_rtp[:, :3] = imf.cartesian_to_polar(xyz[:, :3])
#mask_rtp = (xyz_rtp[:, 0] >= 1e3*start) & (xyz_rtp[:, 0] <= 1e3*end)
# mask_rtp = (xyz_rtp[:, 0] == 70.)

# rtol=3e-3 gives 69.8 <= r <= 70.2 and rtol=4e-3 gives 69.7 <= r <= 70.3
# atol does not seem to make a big difference
# mask_rtp = np.isclose(xyz_rtp[:, 0], 70.*np.ones(xyz_rtp[:, 0].shape), rtol=4e-3, atol=1e-4)
# mask_rtp = mask_rtp & (xyz_rtp[:, 1] <= np.pi/2)
ef_rtp_min, ef_rtp_max = np.min(efield_rtp[:, :3], axis=0), np.max(efield_rtp[:, :3], axis=0)
mask_rtp = (xyz_rtp[:, 0] >= 69.7) & (xyz_rtp[:, 0] <= 70.3) & (xyz_rtp[:, 1] <= np.pi/2)
#mask_rtp = mask_rtp & (xyz_rtp[:, 1] >= ef_rtp_min[1]) & (xyz_rtp[:, 1] <= ef_rtp_max[1])
#mask_rtp = mask_rtp & (xyz_rtp[:, 2] >= ef_rtp_min[2]) & (xyz_rtp[:, 2] <= ef_rtp_max[2])
#xyz_rtp_masked = xyz_rtp[mask_rtp, :]
xyz_rtp_masked = imf.polar_to_cartesian(xyz_rtp[mask_rtp, :])
# Interpolate using griddata, method can be 'linear', 'nearest', 'cubic'
#efield_rtp_interp = griddata(efield_rtp[:, 1:3], efield_rtp[:, 3], xyz_rtp_masked[:, 1:3], method='cubic', fill_value=0.)
efield_rtp_interp = griddata(efield_xyz[:, :3], efield_rtp_extended[:, 3], xyz_rtp_masked[:, :3], method='linear', fill_value=0.)
# efield_rtp_interp_nan = griddata(efield_rtp[:, 1:3], efield_rtp[:, 3], xyz_rtp_masked[:, 1:3], method='cubic', fill_value=np.nan)
# efield_rtp_interp = efield_rtp_interp_nan[np.isnan(efield_rtp_interp_nan)]
# xyz_rtp_masked = xyz_rtp_masked[np.isnan(efield_rtp_interp_nan)]
#xyz_rtp_masked = xyz_rtp_masked[efield_rtp_interp > 0.]
#efield_rtp_interp = efield_rtp_interp[efield_rtp_interp > 0.]
# %%
# compute the efield measurements coordinates in the image space, i.e., ijk indices in the 3D array
efield_ijk = np.transpose(np.linalg.inv(affine) @ efield_xyz.T).astype(np.int64)[:, :3]
efield_ijk -= 1

# select the range of measured efield coordinates in xyz space +- 5 mm to avoid edge effects
efield_xyz_min, efield_xyz_max = np.min(efield_xyz[:, :3], axis=0) - 5., np.max(efield_xyz[:, :3], axis=0) + 5.
efield_xyz_min_max = [n for n in zip(efield_xyz_min, efield_xyz_max)]

# select mri xyz coordinates only within the measured efield volume for faster interpolation
mask_xyz = np.all([((xyz[:, i] >= efield_xyz_min_max[i][0]) & (xyz[:, i] <= efield_xyz_min_max[i][1])) for i in range(xyz.shape[1])], axis=0)

xyz_in_efield_volume = xyz[mask_xyz].copy()

# %%
# Create a grid upon which to interpolate
# grid_t, grid_p = np.meshgrid(xyz_rtp_masked[:, 1], xyz_rtp_masked[:, 2], sparse=True)

# %%
# select mri ijk coordinates only within the measured efield volume for faster interpolation
efield_ijk_min, efield_ijk_max = np.min(efield_ijk, axis=0), np.max(efield_ijk, axis=0)
efield_ijk_min_max = [n for n in zip(efield_ijk_min, efield_ijk_max)]

mask_ef_ijk = np.all([((efield_ijk[:, i] >= efield_ijk_min_max[i][0]) & (efield_ijk[:, i] <= efield_ijk_min_max[i][1])) for i in range(efield_ijk.shape[1])], axis=0)


# %%
# apply the efield value to the corresponding index in the 3D MRI volume (array)
efield_array = np.zeros(img_shape)
for n, ef in enumerate(efield_ijk):
    efield_array[*ef] = efield.values[n, 3]

# %%

#image_coordinates = np.hstack([image_coordinates + 1, np.ones((image_coordinates.shape[0], 1))]).reshape([4, image_coordinates.shape[0]])

# %%
# Assuming z is your 2D array
# Creating x, y coordinates for the 2D array
n_slice = 120

z = image_array[n_slice, :, :]
# Create the Triangulation; no triangles so Delaunay triangulation created.
triang = tri.Triangulation(xyz[:, 1], xyz[:, 2])
triang_mri = tri.Triangulation(image_coordinates[:, 1] - 1., image_coordinates[:, 2] - 1)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False)

# %
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))

tcf = axs[0].tricontourf(triang, z.flatten(), cmap='gray', levels=255)
tcf_mri = axs[1].tricontourf(triang_mri, z.flatten(), cmap='viridis', levels=255)

for ax in axs:
    ax.set_aspect('equal')

# Add a color bar which maps values to colors.
# fig.colorbar(tcf, shrink=0.5, aspect=5)

plt.show()


# %%
#image_coordinates_slice = image_coordinates[:, :5].copy()
# ijk_homo = np.transpose(affine_noscale @ image_coordinates)
xyz_nibabel = np.zeros([3, image_coordinates_homo_slice.shape[0]])
#xyz_mat = np.zeros(image_coordinates_slice.shape)
#xyz_dot = np.zeros(image_coordinates_slice.shape)
for n, coord in enumerate(image_coordinates_homo_slice):
    print("coord: ", coord)
    xyz_nibabel[:, n] = nb.affines.apply_affine(affine, coord[:3])

xyz_mat = np.dot(affine, image_coordinates_homo_slice.T)
xyz_dot = affine @ image_coordinates_homo_slice.T

#print("max: ", np.max(ijk_homo, axis=0))
#print("min: ", np.min(ijk_homo, axis=0))
# %%
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

# %%
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3, 6))
slices = [efield_array[81, ...], efield_array[:, 203, :], efield_array[..., 457]]
for i, slice in enumerate(slices):
    axs[i].imshow(slice.T, cmap="cividis", origin="lower")
plt.show()

# %%
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))
# slices = efield_array[:, :, 457]
axs[0].imshow(efield_array[:, :, 332], cmap="cividis", origin="lower")
axs[1].imshow(image_array[:, :, 450], cmap="gray", origin="lower")
for ax in axs:
    ax.set_aspect(2.)
plt.show()

# %%
