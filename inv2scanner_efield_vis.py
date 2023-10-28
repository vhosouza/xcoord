#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import image_funcs as imf
from visualization import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas\efield'

    filenames = {'T1': 'sub-S1_ses-S8741_T1w', 'HEAD': 'sub-S1_ses-S8741_T1w', 'BRAIN': 'gm',
                 'HEADSIM': 'sub-S1_ses-S8741_T1w', 'BRAINSIM': 'mesh', 'DATA': 'data'}

    # .stl files
    # mesh = not scaled from matlab
    # sub-S1_ses-S8741_T1w = exported from invesalius .inv3 project
    # e_mesh_matlab_stlwrite = scaled in matlab
    # invesalius_cortexmeshlab = scaled in meshlab
    # invesalius_cortexbary_center = origin changed to barycenter in meshlab

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    # head_inv_path = os.path.join(data_dir, filenames['HEAD'] + '.stl')
    brain_inv_path = os.path.join(data_dir, filenames['BRAIN'] + '.stl')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    data_path = os.path.join(data_dir, filenames['DATA'] + '.mat')

    imagedata, affine = imf.load_image(img_path)
    mri2inv_mat = imf.mri2inv(imagedata, affine)
    inv2mri_mat = imf.inv2mri(imagedata, affine)

    data = imf.load_mesh_mat(data_path)

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        ren, ren_win, iren = vf.create_window()

        _ = vf.create_mesh(data, [0., 1., 0.], ren)

        # 0: red, 1: green, 2: blue, 3: maroon (dark red),
        # 4: purple, 5: teal (petrol blue), 6: yellow, 7: orange
        colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
                   [0.45, 0., 0.5], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

        repos = [0., 0., 0., 0., 0., 0.]

        # _ = vf.load_stl(head_sim_path, ren, opacity=.4, colour=[0.482, 0.627, 0.698], replace=repos, user_matrix=np.identity(4))
        # _ = vf.load_stl(head_sim_path, ren, opacity=1., colour="SkinColor", replace=repos, user_matrix=np.identity(4))
        _ = vf.load_stl(head_sim_path, ren, opacity=1., colour="SkinColor", replace=repos, user_matrix=inv2mri_mat)
        _ = vf.load_stl(brain_inv_path, ren, opacity=.5, colour=[1., 0., 0.], replace=repos, user_matrix=np.identity(4))

        # _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos, user_matrix=np.identity(4))
        # _ = vf.load_stl(brain_sim_path, ren, opacity=.5, colour=[1., 1., 1.], replace=repos, user_matrix=np.identity(4),
        #                 scale=1000)

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
