#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np

import image_funcs as imf
import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True
    COMP_ACT_SEED = True

    SEED_OFFSET = 25

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'
    # markers_20210304_all_before_rep_scan
    filenames = {'MKSS': 'markers_20210304_rep_left_m1_dlpfc_broca_V1_final_scan_seed', 'COIL': 'magstim_fig8_coil',
                 'BRAINSIM': 'wm', 'HEADSIM': 'skin', 'ACT': 'trekkerACTlabels',
                 'T1': 'sub-S1_ses-S8741_T1w'}

    mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    coil_path = os.path.join(data_dir, filenames['COIL'] + '.stl')
    act_path = os.path.join(data_dir, filenames['ACT'] + '.nii')
    # img_path = os.path.join(data_dir, filenames['T1'] + '.nii')

    coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)

    id_fids = ['LEI', 'REI', 'NAI']
    index_coord = id_list.index('v1-right')
    index_fids = [id_list.index(n) for n in id_fids]

    if COMP_ACT_SEED:
        imagedata, affine = imf.load_image(act_path)
        # imagedata2, affine2 = imf.load_image(img_path)
        img_shape = imagedata.header.get_data_shape()
        act_data = imagedata.get_fdata()
        mri2inv_mat = imf.mri2inv(imagedata, affine)

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        ren, ren_win, iren = vf.create_window()

        # 0: red, 1: green, 2: blue, 3: maroon (dark red),
        # 4: purple, 5: teal (petrol blue), 6: yellow, 7: orange
        colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
                   [0.45, 0., 0.5], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

        repos = [0., 0., 0., 0., 0., 0.]

        _ = vf.load_stl(head_sim_path, ren, opacity=.4, colour="SkinColor", replace=repos,
                     user_matrix=np.identity(4))

        _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos,
                     user_matrix=np.identity(4))

        # if COMP_ACT_SEED:
        #     _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos,
        #                     user_matrix=np.linalg.inv(affine))
        #     _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 0.], replace=repos,
        #                     user_matrix=mri2inv_mat)

        # create fiducial markers
        for n in index_fids:
            _ = vf.add_marker(coord_list[n], ren, colours[n], radius=2)
        # --- fiducial markers

        # create coil vectors
        coil_pos = np.hstack((coord_list[index_coord], orient_list[index_coord]))
        m_coil = imf.coil_transform_matrix(coil_pos)

        vec_length = 75
        repos_coil = [0., 0., 0., 0., 0., 90.]

        # coil vectors in invesalius 3D space
        p1 = m_coil[:-1, -1]
        coil_dir = m_coil[:-1, 0]
        coil_face = m_coil[:-1, 1]
        p2_face = p1 + vec_length * coil_face
        p2_dir = p1 + vec_length * coil_dir
        coil_norm = np.cross(coil_dir, coil_face)
        p2_norm = p1 - vec_length * coil_norm

        if COMP_ACT_SEED:
            coord_list_w = imf.create_grid((-2, 2), (0, 20), SEED_OFFSET - 5, 1)
            coord_list_w_tr = m_coil @ coord_list_w
            coord_offset = imf.grid_offset(act_data, coord_list_w_tr, affine=affine)
            # coord_list_w_tr_inv = mri2inv_mat @ m_coil @ coord_list_w
            # coord_list_inv = imf.grid_offset_inv(act_data, coord_list_w_tr_inv, img_shape[1])
            # coord_list_mri = np.squeeze(coord_list_inv + np.array([[0, img_shape[1], 0]]))

        # offset = 40
        # coil_norm = coil_norm/np.linalg.norm(coil_norm)
        # coord_offset = p1 - offset * coil_norm

        # _ = vf.load_stl(coil_path, ren, opacity=.6, replace=repos_coil, colour=[1., 1., 1.], user_matrix=m_coil)

        _ = vf.add_line(ren, p1, p2_dir, color=[1.0, .0, .0])
        _ = vf.add_line(ren, p1, p2_face, color=[.0, 1.0, .0])
        _ = vf.add_line(ren, p1, p2_norm, color=[.0, .0, 1.0])

        _ = vf.add_marker(p1, ren, colours[4], radius=2)

        # --- coil vectors

        # seed markers
        _ = vf.add_marker(seed_list[index_coord], ren, colours[5], radius=.5)
        _ = vf.add_marker(coord_offset, ren, colours[6], radius=.5)
        # _ = vf.add_marker(coord_list_inv, ren, colours[0], radius=.5)
        # _ = vf.add_marker(coord_list_mri, ren, colours[0], radius=.5)

        # for n in coord_list_w_tr.T:
        #     _ = vf.add_marker(n[:3], ren, colours[7], radius=.5, opacity=.2)
        # --- seed markers

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
