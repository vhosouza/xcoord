#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import sys

from scipy.interpolate import griddata

import external.transformations as tf
import image_funcs

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

tms_system = 'nexstim'

data_dir = os.path.join(onedrive_path, 'projects', 'nexstim', 'data', 'mri')
filenames = 'GJ_2008_anonym_t1_mpr_ns_sag_1_1_1_mm_20081021180940_2'
filename_efield_extended = f'{tms_system}_efield_map_1000_extended'
filename_efield = f'{tms_system}_efield_map_1000_rtp'

filepaths = os.path.join(data_dir, f'{filenames}.nii.gz')
filepath_out_new_img = os.path.join(data_dir, f'{tms_system}_efield_map.nii.gz')
filepath_out_png = os.path.join(data_dir, f'{tms_system}_efield_mri_overlay.png')
#filepath_out_png = os.path.join(data_dir, f'{tms_system}_efield_mri_overlay_colorbar.png')
filepath_efield = os.path.join(data_dir, f'{filename_efield}.csv')
filepath_efield_interp = os.path.join(data_dir, f'{filename_efield}_interp.npz')

efield = pd.read_csv(filepath_efield, delimiter=';')
efield_rtp = efield.to_numpy(copy=True)
efield_rtp[:, 0] *= 1e3
efield_rtp_min, efield_rtp_max = np.min(efield_rtp[:, :3], axis=0), np.max(efield_rtp[:, :3], axis=0)

imagedata = nb.squeeze_image(nb.load(filepaths))
imagedata = nb.as_closest_canonical(imagedata)
imagedata.update_header()
pix_dim = imagedata.header.get_zooms()
image_shape = imagedata.header.get_data_shape()
image_array = imagedata.get_fdata()
affine = imagedata.affine

# %
# Create the image coordinates in xyz and rtp
# Get the indices
image_indices = np.indices(image_shape)
# Transpose and reshape to get triplets
image_ijk = np.transpose(image_indices).reshape(-1, 3).astype(np.float64) + 1.
image_ijk = np.hstack((image_ijk, np.ones((image_ijk.shape[0], 1))))

# compute 3D array MRI indices coordinates in worlds/scanner/xyz space
image_xyz = np.transpose(affine @ image_ijk.T)[:, :3]
image_xyz_min, xyz_max = np.min(image_xyz[:, :3], axis=0), np.max(image_xyz[:, :3], axis=0)
image_xyz_min_max = [n for n in zip(image_xyz_min, xyz_max)]

image_rtp = imf.cartesian_to_polar(image_xyz.copy())
# select upper hemisphere
image_rtp_hem = image_rtp[image_rtp[:, 1] <= np.pi/2].copy()

# %
# Extend the measured E-field within a range of radius
# radius_start, radius_end, dr = 67.25, 74.0, 0.25
radius_start, radius_end, dr = 69.25, 72.0, 0.25
r_range = np.arange(radius_start, radius_end, dr)
efield_rtp_extended = []
for r in r_range:
    efield_rtp_aux = efield_rtp.copy()
    efield_rtp_aux[:, 0] = r
    efield_rtp_extended.append(efield_rtp_aux)

efield_rtp_extended = np.vstack(efield_rtp_extended)
efield_xyz_extended = efield_rtp_extended.copy()
efield_xyz_extended[:, :3] = imf.polar_to_cartesian(efield_rtp_extended[:, :3])

# %
# Interpolate 3D using griddata, method can be 'linear' or 'nearest'
# select coordinates close to the radius within a range defined by
# (r +- dr) and 0 <= theta <= 90 deg
# r_range = np.arange(radius_start + dr, radius_end, 2*dr)
# efield_interp = []
# image_xyz_interp = []
# for r in r_range:
#     mask_rtp = (image_rtp_hem[:, 0] >= r - dr) & (image_rtp_hem[:, 0] <= r + dr)
#     image_xyz_masked = imf.polar_to_cartesian(image_rtp_hem[mask_rtp, :])
#     efield_interp.append(griddata(efield_xyz_extended[:, :3], efield_xyz_extended[:, 3], image_xyz_masked[:, :3],
#                                   method='linear', fill_value=0.))
#     image_xyz_interp.append(image_xyz_masked[:, :3])
#     print(f"Done interp for radius: {r} out of {radius_end - 2*dr}.")
#
# efield_interp = np.vstack([n[:, np.newaxis] for n in efield_interp])
# image_xyz_interp = np.vstack(image_xyz_interp)
#
# # Save efield interp
# np.savez(filepath_efield_interp, efield_interp=efield_interp, image_xyz_interp=image_xyz_interp)
# %%
# Load efield interp
efield_interp_file = np.load(filepath_efield_interp)
efield_interp = efield_interp_file['efield_interp']
image_xyz_interp = efield_interp_file['image_xyz_interp']
del efield_interp_file
# %%
image_rtp_interp = imf.cartesian_to_polar(image_xyz_interp)
mask_r = (image_rtp_interp[:, 0] > 69.5) & (image_rtp_interp[:, 0] < 71.5)
mask_r_theta = mask_r & (image_rtp_interp[:, 1] < np.deg2rad(90))
mask_r_theta_intensity = mask_r_theta & (np.squeeze(efield_interp > 0.7*efield_interp.max()))
efield_interp = efield_interp[mask_r_theta_intensity]
image_xyz_interp = image_xyz_interp[mask_r_theta_intensity]

dlpfc_ijk = np.array(((49), (image_shape[2] - 198), (388), (1))).reshape(4, 1)
dlpfc_xyz = np.transpose(affine @ dlpfc_ijk)[:, :3]
coil_xyz = np.array([[-51.22, 65.37, 49.90]])
coil_ijk = np.transpose(np.linalg.inv(affine) @ np.vstack((coil_xyz.T, [1])))[:, :3].astype(np.int64)
v_coil_dlpfc = np.squeeze(coil_xyz - dlpfc_xyz)
# %%
# compute the efield measurements coordinates in the image space, i.e., ijk indices in the 3D array
image_xyz_interp_homo = np.hstack((image_xyz_interp, np.ones((image_xyz_interp.shape[0], 1))))
#translate, euler_angles = (-43.5, 65.5, -34.), np.deg2rad((-55., 0., 0.))
#translate, euler_angles = (-43.5, 30.5, -14.), np.deg2rad((-0., 0., 0.))
# this is almost good
#translate, euler_angles = (10.5, 20.5, -4.), np.deg2rad((-60., -15., 45.))
#translate, euler_angles = (0., 0., 0.), np.deg2rad((-60., -15., 0.))

# -55, -45, 0 # looked quite good
rx = tf.rotation_matrix(np.deg2rad(-45), [1, 0, 0])
ry = tf.rotation_matrix(np.deg2rad(-30), [0, 1, 0])
rz = tf.rotation_matrix(np.deg2rad(25), [0, 0, 1])
raround = tf.rotation_matrix(np.deg2rad(-30), v_coil_dlpfc)

# transform_efield = tf.compose_matrix(scale=None, shear=None, angles=euler_angles, translate=translate, perspective=None)
# translate_to_origin = tf.translation_matrix(-ef_xyz_max)
transform_efield = raround @ rz @ rx @ ry

ef_xyz_max = image_xyz_interp_homo[efield_interp.argmax(), :3]
translate_to_origin = tf.translation_matrix(-ef_xyz_max)
translate_to_dlpfc = tf.translation_matrix(dlpfc_xyz)

image_xyz_interp_homo_transformed = translate_to_dlpfc @ transform_efield @ translate_to_origin @ image_xyz_interp_homo.T

# %%
# image_xyz_interp_transformed = image_xyz_interp_homo_transformed.T[:, :3]
# start_indices, end_indices = np.min(image_xyz_interp_transformed, axis=0), np.max(image_xyz_interp_transformed, axis=0)
# masks = [(image_xyz[:, n] >= start_indices[n]) & (image_xyz[:, n] <= end_indices[n]) for n in range(3)]
# mask_xyz = masks[0] & masks[1] & masks[2]
# image_xyz_masked_2 = image_xyz[mask_xyz]

image_xyz_interp_transformed = image_funcs.cartesian_to_polar(image_xyz_interp_homo_transformed.T[:, :3], offset_phi=0.)
start_indices, end_indices = np.min(image_xyz_interp_transformed, axis=0), np.max(image_xyz_interp_transformed, axis=0)
masks = [(image_rtp[:, n] >= start_indices[n]) & (image_rtp[:, n] <= end_indices[n]) for n in range(3)]
mask_xyz = masks[0] & masks[1] & masks[2]
image_xyz_masked_2 = image_rtp[mask_xyz]

efield_interp_transformed = griddata(image_xyz_interp_transformed, efield_interp[:, 0], image_xyz_masked_2,
                                     method='linear', fill_value=0.)

# %%
# image_ijk_interp = np.transpose(np.linalg.inv(affine) @ image_xyz_interp_homo.T).astype(np.int64)[:, :3]
# was used for the first fully working version but with the problem of empty pixels after transformation
# image_ijk_interp = np.transpose(np.linalg.inv(affine) @ image_xyz_interp_homo_transformed).astype(np.int64)[:, :3]
# image_xyz_masked_2_homo = np.vstack((image_xyz_masked_2.T, np.ones((1, image_xyz_masked_2.shape[0]))))
image_xyz_masked_2_homo = np.vstack((image_funcs.polar_to_cartesian(image_xyz_masked_2).T, np.ones((1, image_xyz_masked_2.shape[0]))))
image_ijk_interp = np.transpose(np.linalg.inv(affine) @ image_xyz_masked_2_homo).astype(np.int64)[:, :3]
image_ijk_interp -= 1

# image_ijk_interp = np.linalg.inv(affine) @ image_xyz_interp_homo.T
# image_ijk_interp[:3, :] -= 1

# ef_ijk_max = image_ijk_interp[:3, efield_interp.argmax()].T
# translate_to_origin = tf.translation_matrix(-ef_ijk_max)
# translate_to_dlpfc = tf.translation_matrix((49, image_shape[2] - 198, 388))
# transform_efield = translate_to_dlpfc @ rz @ rx @ ry @ translate_to_origin
# #rxyz = tf.euler_matrix(np.deg2rad(-55), np.deg2rad(-20), np.deg2rad(30), axes='ryxy')
# rxyz = tf.euler_matrix(np.deg2rad(0), np.deg2rad(-30), np.deg2rad(-20), axes='rzyx')
# rxyz = tf.euler_matrix(np.deg2rad(0), np.deg2rad(-45), np.deg2rad(0), axes='sxyz')
# transform_efield = translate_to_dlpfc @ rxyz @ translate_to_origin
#
# image_ijk_interp = np.transpose(transform_efield @ image_ijk_interp).astype(np.int64)[:, :3]

# %%
# apply the efield value to the corresponding index in the 3D MRI volume (array)
efield_array = np.zeros(image_shape)
count = 0
for n, ijk in enumerate(image_ijk_interp):
    try:
        # was used for the first fully working version but with the problem of empty pixels after transformation
        # efield_array[*ijk] = efield_interp[n, 0]
        efield_array[*ijk] = efield_interp_transformed[n]
    except IndexError:
        count += 1
        print(f"Voxel {ijk} out of bounds.")

print(f"Total voxels unassigned: {count} out of {n + 1}.")

# %
#fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(8, 4))

# Normalize data1 to be in range(0,1) for an alpha array
#data_norm = (efield_array[:, 150, :].T - np.min(efield_array[:, 150, :].T)) / np.ptp(efield_array[:, 150, :].T)
# Use normalized data as alpha (transparency) and reverse it with 1 - data_norm
#rgba_colors = colors.to_rgba(efield_array[:, 150, :].T, alpha=1 - data_norm)
n_slices = (49, image_shape[2] - 198, 388)
# sagital is , coronal is xz, axial is xy
dlpfc = ((image_shape[2] - 198, 388), (49, 388), (49, image_shape[2] - 198))
#coil_ijk = np.array([[ 35, 325, 415]])
#coil = ((327+14, 418+14), (34-3, 418+14), (34-3, 327+14))

min_val, max_val = np.min(efield_array), np.max(efield_array)
mid_val = (min_val + max_val) / 2.0
efield_slices = [efield_array[n_slices[0], :, :].T, efield_array[:, n_slices[1], :].T, efield_array[:, :, n_slices[2]].T]
mri_slices = [image_array[n_slices[0], :, :].T, image_array[:, n_slices[1], :].T, image_array[:, :, n_slices[2]].T]
#efield_norm = (efield_array - min_val) / (max_val - min_val)
efield_norm = efield_array.copy()
efield_norm[efield_norm > 1] = 1.
efield_norm_slices = [efield_norm[n_slices[0], :, :].T, efield_norm[:, n_slices[1], :].T, efield_norm[:, :, n_slices[2]].T]

# Define a new colormap norm
cmap_norm = colors.Normalize(vmin=mid_val, vmax=max_val, clip=True)
cmap = cm.plasma
aspect_list = [1., .5, .5]

fig = plt.figure(figsize=(8, 4))
fig.patch.set_facecolor('black')
gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1., .7, .7])
axs = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

count = 0
for data_efield, data_mri, data_norm in zip(efield_slices, mri_slices, efield_norm_slices):
    ax = axs[count]
    rgba_img = cmap(cmap_norm(data_efield))
    #rgba_img[..., 3] = data_norm
    rgba_img[..., 3] = data_norm

    ax.imshow(data_mri, cmap="gray", origin="lower", alpha=1, aspect='auto')
    im_ef = ax.imshow(rgba_img, origin="lower", alpha=0.7, aspect='auto', cmap='plasma')
    ax.scatter(dlpfc[count][0], dlpfc[count][1], c='cyan', marker='o', s=20, edgecolor='k', linewidths=0.5)
    #ax.scatter(coil[count][0], coil[count][1], c='cyan', marker='o', s=5)

    #ax.set_xlim(0, 150)
    #ax.set_ylim(0, 150)

    ax.set_aspect(aspect_list[count])
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    count += 1

#fig.colorbar(im_ef, ax=ax)

fig.tight_layout()
#plt.show()

plt.savefig(filepath_out_png, dpi=300, format='png', bbox_inches='tight')

# %%
# plot efield in 3d
# efield_points = 1e3*imf.polar_to_cartesian(efield_rtp[:, :3])
# xp, yp, zp = efield_points[:, 0], efield_points[:, 1], efield_points[:, 2]
# up = efield_rtp[:, 3]
# #u = efield_rtp[:, 3]
# #efield_xyz = imf.polar_to_cartesian(xyz_rtp_masked[:, :3])
# #efield_xyz_plot = image_xyz_interp[:, :3]
# #u = efield_interp
# efield_xyz_plot = image_ijk_interp[efield_interp_transformed > 1, :3]
# u = efield_interp_transformed[efield_interp_transformed > 1]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = efield_xyz_plot[:, 0], efield_xyz_plot[:, 1], efield_xyz_plot[:, 2]
# # ef_max = np.squeeze(efield_interp > 0.97*efield_interp.max())
# ef_max = np.squeeze(efield_interp_transformed[efield_interp_transformed > 1] > 0.97*efield_interp_transformed.max())
# xmax, ymax, zmax = efield_xyz_plot[ef_max, 0], efield_xyz_plot[ef_max, 1], efield_xyz_plot[ef_max, 2]
# # norm_p = colors.Normalize(up.min(), up.max())
# # colormap_p = cm.copper
# # mappable_p = cm.ScalarMappable(norm=norm_p, cmap=colormap_p)
# # mappable_p.set_array(up)
#
# norm = colors.Normalize(u.min(), u.max())
# colormap = cm.plasma
# mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
# mappable.set_array(u)
# #surf = ax.plot_trisurf(x, y, z, facecolors=mappable.to_rgba(u), cmap=colormap, linewidth=1, shade=False)
# #sc = ax.scatter3D(xp, yp, zp, facecolors=mappable_p.to_rgba(up), cmap=colormap_p)
# sc = ax.scatter3D(x[::50], y[::50]/2, z[::50]/2, facecolors=mappable.to_rgba(u[::50]), alpha=0.1)
# sc = ax.scatter3D(49, 314/2, 388/2, color='m', marker='o', s=15)
# sc = ax.scatter3D(35, (512-186)/2, 415/2, color='k', marker='o', s=15)
# #sc = ax.scatter3D(xmax, ymax/2, zmax/2, color='k', marker='o', s=15)
# sc = ax.plot(np.linspace(0, image_shape[0], 100),
#              np.zeros(100) + 314/2,
#              np.zeros(100) + 388/2, '-r')
# sc = ax.plot(np.zeros(100) + 49,
#              np.linspace(0, image_shape[1]/2, 100),
#              np.zeros(100) + 388/2, '-b')
# sc = ax.plot(np.zeros(100) + 49,
#              np.zeros(100) + 314/2,
#              np.linspace(0, image_shape[2]/2, 100), '-g')
# elevation_angle = 42  # Up/down
# azimuth_angle = 155  # Left/right # 120 good to see the dlpfc
# ax.view_init(elev=elevation_angle, azim=azimuth_angle)
# #ax.set_aspect('equal')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(0, image_shape[0])
# ax.set_ylim(0, image_shape[1]/2)
# ax.set_zlim(0, image_shape[2]/2)
# fig.tight_layout()
# plt.show()

# %%
# Now create a new NIfTI image with the new data. It's important to keep the old header.
new_img = nb.Nifti1Image(efield_array, imagedata.affine, imagedata.header)
# Save the new image to disk

nb.save(new_img, filepath_out_new_img)

# %
# fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(4, 6))
# slices = [efield_array[81, ...], efield_array[:, 203, :], efield_array[..., 457]]
# for i, slice in enumerate(slices):
#     axs[i].imshow(slice.T, cmap="cividis", origin="lower")
# plt.show()
#
# # %%
# fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 6))
# # slices = efield_array[:, :, 457]
# axs[0].imshow(efield_array[:, 150, :].T, cmap="cividis", origin="lower")
# axs[1].imshow(image_array[:, 150, :].T, cmap="gray", origin="lower")
#
# for ax in axs:
#     ax.set_aspect(.5)
#     for edge, spine in ax.spines.items():
#         spine.set_visible(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()
#
# # %%
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 6))
#
# # Normalize data1 to be in range(0,1) for an alpha array
# #data_norm = (efield_array[:, 150, :].T - np.min(efield_array[:, 150, :].T)) / np.ptp(efield_array[:, 150, :].T)
# # Use normalized data as alpha (transparency) and reverse it with 1 - data_norm
# #rgba_colors = colors.to_rgba(efield_array[:, 150, :].T, alpha=1 - data_norm)
# n_slice = image_shape[2] - 198
# data_efield = efield_array[:, n_slice, :].T
# # Define a new colormap norm
# min_val = np.min(data_efield)
# max_val = np.max(data_efield)
# mid_val = (min_val + max_val) / 2.0
# cmap_norm = colors.Normalize(vmin=mid_val, vmax=max_val, clip=True)
#
# data_norm = (data_efield - np.min(data_efield)) / np.ptp(data_efield.T)
# cmap = cm.autumn
# rgba_img = cmap(cmap_norm(data_efield))
# rgba_img[..., 3] = data_norm * 0.9
#
# ax.imshow(image_array[:, n_slice, :].T, cmap="gray", origin="lower", alpha=1)
# #ax.imshow(efield_array[:, 150, :].T, cmap="cividis", origin="lower", alpha=1 - data1_norm)
# ax.imshow(rgba_img, origin="lower")
#
# ax.set_aspect(.5)
# for edge, spine in ax.spines.items():
#     spine.set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])
# fig.tight_layout()
# # plt.savefig(filepath_out_png, dpi=300, format='png', bbox_inches='tight')
# plt.show()

