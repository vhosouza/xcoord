#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
from scipy.io import savemat

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

    # %%
    data_dir = r'P:\tms_eeg\mTMS\projects\2019 EEG-based target automatization\Analysis\EEG electrode transformation\Locations of interest in Nexstim coords'

    filenames = {'FILE_LIST': 'EEGTA_subjects_T1_MRI_paths'}

    file_list_path = os.path.join(data_dir, filenames['FILE_LIST'] + '.csv')
    with open(file_list_path, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=";")
        file_list = [data for data in data_iter]
    # file_list = np.asarray(data)
    # remove header
    file_list.pop(0)

    eeg_path_list = [os.path.join(s[4], s[5]) for s in file_list]
    img_path_list = [os.path.join(s[2], s[3]) for s in file_list]
    subj_list = [s[0] for s in file_list]

    # %%
    for subj_id in range(len(subj_list)):
        coord_list = imf.load_eeg_mat(eeg_path_list[subj_id], variable='POI_coords_Nexstim')
        n_points = coord_list.shape[0]

        imagedata, affine = imf.load_image(img_path_list[subj_id], closest=True)
        img_shape = imagedata.header.get_data_shape()

        coords_np = np.zeros([n_points, 3])
        for n, coord in enumerate(coord_list):
            # the flipx and reorder are needed to transform from nexstim to mri space
            coords_np[n, :] = n2m.coord_change(coord, img_shape, affine, flipx, reorder)
            # export the coordinates changed only by the affine
            # coords_np[n, :] = n2m.coord_change(coord, img_shape, affine)

        if SAVE_EEG:
            save_eeg_path = os.path.splitext(eeg_path_list[subj_id])[0] + '_MRI.csv'
            save_eeg_mat_path = os.path.splitext(eeg_path_list[subj_id])[0] + '_MRI.mat'
            np.savetxt(save_eeg_path, coords_np, delimiter=";")
            mat_dic = {'coords_mri': coords_np}
            savemat(save_eeg_mat_path, mat_dic)

    # %%
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
