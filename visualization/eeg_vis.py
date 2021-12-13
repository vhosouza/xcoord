#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import image_funcs as imf
from markers import nexstim2mri as n2m
from visualization import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True
    SHOW_EEG = True
    SAVE_EEG = False

    reorder = [0, 2, 1]
    flipx = [True, False, False]

    data_dir = r'P:\tms_eeg\mTMS\projects\2019 EEG-based target automatization\Analysis\EEG electrode transformation\Locations of interest in Nexstim coords'

    filenames = {'T1': 'EEGTA04', 'EEG': 'EEGTA04_electrode_locations_Nexstim',
                 'TARGET': 'EEGTA04_preSMA_target', 'SAVE_EEG': 'EEGTA04_electrode_locations_MRI'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    eeg_path = os.path.join(data_dir, filenames['EEG'] + '.mat')
    target_path = os.path.join(data_dir, filenames['TARGET'] + '.mat')
    save_eeg_path = os.path.join(data_dir, filenames['SAVE_EEG'] + '.csv')
    # brain_inv_path = os.path.join(data_dir, filenames['BRAIN'] + '.stl')
    # head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    # brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    # coil_path = os.path.join(data_dir, filenames['COIL'] + '.stl')

    coord_list = imf.load_eeg_mat(eeg_path)
    n_points = coord_list.shape[0]

    imagedata, affine = imf.load_image(img_path, closest=True)
    img_shape = imagedata.header.get_data_shape()
    # mri2inv_mat = imf.mri2inv(imagedata, affine)
    # inv2mri_mat = imf.inv2mri(imagedata, affine)

    coords_np = np.zeros([n_points, 3])
    for n, coord in enumerate(coord_list):
        # the flipx and reorder are needed to transform from nexstim to mri space
        coords_np[n, :] = n2m.coord_change(coord, img_shape, affine, flipx, reorder)
        # export the coordinates changed only by the affine
        # coords_np[n, :] = n2m.coord_change(coord, img_shape, affine)

    if SAVE_EEG:
        np.savetxt(save_eeg_path, coords_np, delimiter=";")

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        ren, ren_win, iren = vf.create_window()

        if SHOW_EEG:
            for coord in coords_np:
                _ = vf.add_marker(coord, ren, [1., 0., 0.], radius=2)

        # Add axes to scene origin
        if SHOW_AXES:
            _ = vf.add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
            _ = vf.add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
            _ = vf.add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

        # Initialize window and interactor
        iren.Initialize()
        ren_win.Render()
        iren.Start()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.set_printoptions(suppress=True, precision=2)
    main()
