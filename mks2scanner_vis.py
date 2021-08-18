#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import image_funcs as imf
import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'

    filenames = {'T1': 'sub-S1_ses-S8741_T1w', 'MKSS': 'markers_20210304_m1',
                 'ACT': 'trekkerACTlabels', 'COIL': 'magstim_fig8_coil',
                 'HEAD': 'head_inv', 'BRAIN': 'brain_inv', 'BRAINSIM': 'wm',
                 'HEADSIM': 'skin', 'MKS': 'ana_markers3'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    mkss_path = os.path.join(data_dir, filenames['MKS'] + '.mks')
    # mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    # head_inv_path = os.path.join(data_dir, filenames['HEAD'] + '.stl')
    # brain_inv_path = os.path.join(data_dir, filenames['BRAIN'] + '.stl')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    coil_path = os.path.join(data_dir, filenames['COIL'] + '.stl')

    coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)
    n_points = coord_list.shape[0]

    coord_w = np.hstack((coord_list, np.ones((n_points, 1))))
    seed_w = np.hstack((seed_list, np.ones((n_points, 1))))

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

        _ = vf.load_stl(head_sim_path, ren, opacity=.4, colour=[0.482, 0.627, 0.698], replace=repos, user_matrix=mri2inv_mat)
        _ = vf.load_stl(head_sim_path, ren, opacity=.4, colour="SkinColor", replace=repos,
                     user_matrix=np.identity(4))

        _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos, user_matrix=mri2inv_mat)
        _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos,
                     user_matrix=np.identity(4))

        # create fiducial markers
        fids_inv_vtk = coord_w[:3, :].copy()
        # from the invesalius exported fiducial markers you have to multiply the Y coordinate by -1 to
        # transform to the regular 3D invesalius space where coil location is saved
        fids_inv_vtk[:, 1] *= -1
        fids_vis = fids_inv_vtk[:, :3]
        fids_scan = inv2mri_mat @ fids_inv_vtk.T
        fids_scan_vis = fids_scan.T[:3, :3]

        for n in range(3):
            _ = vf.add_marker(fids_scan_vis[n, :], ren, colours[n], radius=2)

        for n in range(3):
            _ = vf.add_marker(fids_vis[n, :], ren, colours[n], radius=2)
        # --- fiducial markers

        # create coil vectors
        coil_pos = np.hstack((coord_w[-1, :3], orient_list[-1, :]))
        coil_pos[1] *= -1
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

        # offset = 40
        # coil_norm = coil_norm/np.linalg.norm(coil_norm)
        # coord_offset_nav = p1 - offset * coil_norm

        _ = vf.load_stl(coil_path, ren, opacity=.6, replace=repos_coil, colour=[1., 1., 1.], user_matrix=m_coil)

        _ = vf.add_line(ren, p1, p2_dir, color=[1.0, .0, .0])
        _ = vf.add_line(ren, p1, p2_face, color=[.0, 1.0, .0])
        _ = vf.add_line(ren, p1, p2_norm, color=[.0, .0, 1.0])

        _ = vf.add_marker(p1, ren, colours[4], radius=2)

        # coil vectors in MRI space
        m_coil_scan = inv2mri_mat @ m_coil
        p1_scan = m_coil_scan[:-1, -1]
        coil_dir_scan = m_coil_scan[:-1, 0]
        coil_face_scan = m_coil_scan[:-1, 1]
        p2_face_scan = p1_scan + vec_length * coil_face_scan
        p2_dir_scan = p1_scan + vec_length * coil_dir_scan
        coil_norm_scan = np.cross(coil_dir_scan, coil_face_scan)
        p2_norm_scan = p1_scan - vec_length * coil_norm_scan

        _ = vf.load_stl(coil_path, ren, opacity=.6, replace=repos_coil, colour=[1., 1., 1.], user_matrix=m_coil_scan)

        _ = vf.add_line(ren, p1_scan, p2_dir_scan, color=[1.0, .0, .0])
        _ = vf.add_line(ren, p1_scan, p2_face_scan, color=[.0, 1.0, .0])
        _ = vf.add_line(ren, p1_scan, p2_norm_scan, color=[.0, .0, 1.0])

        _ = vf.add_marker(p1_scan, ren, colours[4], radius=2)
        # --- coil vectors

        # seed markers
        seed_3dinv = seed_w[np.newaxis, -1, :].copy()
        seed_vis = seed_3dinv[0, :3]
        seed_scan = inv2mri_mat @ seed_3dinv.T
        seed_scan_vis = seed_scan.T[0, :3]

        _ = vf.add_marker(seed_vis, ren, colours[5], radius=2)
        _ = vf.add_marker(seed_scan_vis, ren, colours[6], radius=2)
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
