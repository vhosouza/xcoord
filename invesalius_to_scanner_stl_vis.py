#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import image_funcs as imf
from visualization import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True

    # data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'

    # filenames = {'T1': 'sub-S1_ses-S8741_T1w', 'MKSS': 'markers_20210304_m1',
    #              'ACT': 'trekkerACTlabels', 'COIL': 'magstim_fig8_coil',
    #              'HEAD': 'half_head_world', 'BRAIN': 'brain_inv', 'BRAINSIM': 'wm',
    #              'HEADSIM': 'skin', 'MKS': 'ana_markers3'}

    subj_id = 7
    data_dir = os.environ.get('OneDrive') + r'\data\nexstim_coord\mri\ppM1_S{}'.format(subj_id)

    filenames = {'T1': 'ppM1_S{}'.format(subj_id),
                 'HEAD': 'ppM1_S{}_scalp_shell_world'.format(subj_id), 'HEADSIM': 'ppM1_S{}_brain_shell_world'.format(subj_id),
                 'BRAIN': 'brain_inv', 'BRAINSIM': 'wm'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    head_inv_path = os.path.join(data_dir, filenames['HEAD'] + '.stl')
    # brain_inv_path = os.path.join(data_dir, filenames['BRAIN'] + '.stl')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    # brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')

    imagedata, affine = imf.load_image(img_path)
    mri2inv_mat = imf.mri2inv(imagedata, affine)
    inv2mri_mat = imf.inv2mri(imagedata, affine)

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        ren, ren_win, iren = vf.create_window()

        # 0: red, 1: green, 2: blue, 3: maroon (dark red),
        # 4: purple, 5: teal (petrol blue), 6: yellow, 7: orange
        colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
                   [0.45, 0., 0.5], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

        repos = [0., 0., 0., 0., 0., 0.]

        _ = vf.load_stl(head_inv_path, ren, opacity=.4, colour=[0.482, 0.627, 0.698], replace=repos,
                        user_matrix=np.identity(4))
        _ = vf.load_stl(head_sim_path, ren, opacity=0.4, colour="SkinColor", replace=repos,
                     user_matrix=np.identity(4))

        # _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos, user_matrix=mri2inv_mat)
        # _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos,
        #              user_matrix=np.identity(4))

        pix_dim = 0.48828125
        img_shape = 512
        offset = pix_dim * (img_shape - 1)

        # _ = vf.add_marker([31.25201835082943, 69, -25.883510303223446], ren, colours[0], radius=2)
        _ = vf.add_marker([-2.8687858387879146, 102.87724194445669, -62.21162745582136], ren, colours[1], radius=1)

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
