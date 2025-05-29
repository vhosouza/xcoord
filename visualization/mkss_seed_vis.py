#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import image_funcs as imf
from visualization import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True
    TRANFORM_SCANNER = True
    SAVE_SCANNER = False

    if sys.platform == "win32":
        onedrive_path = os.environ.get('OneDrive')
    elif (sys.platform == "darwin") or (sys.platform == "linux"):
        onedrive_path = os.path.expanduser('~/OneDrive - Aalto University')
    else:
        onedrive_path = False
        print("Unsupported platform")

    subject_id = 0
    subject = ['joonas', 'pantelis', 'baran', 'victor']
    marker_file = {'joonas': '20211123-200911-T1-markers_all_repeated',
                   'pantelis': '20211126-121334-T1-markers_all_repeated',
                   'baran': '20211123-205329-T1-markers_all_repeated',
                   'victor': '20211126-132408-T1-markers_all_repeated'}

    target_names = ['M1', 'V1', 'BROCA', 'DLPFC']
    columns_export = ['label', 'x_seed', 'y_seed', 'z_seed', 'is_target']

    root_dir = os.path.join(onedrive_path, 'data', 'dti_navigation', 'normMRI')
    data_dir = os.path.join(root_dir, subject[subject_id])

    filenames = {'MKSS': marker_file[subject[subject_id]],
                 'COIL': 'magstim_fig8_coil', 'T1': 'T1',
                 'BRAINSIM': 'wm', 'HEADSIM': 'skin'}

    # mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    coil_path = os.path.join(root_dir, filenames['COIL'] + '.stl')
    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    save_path = os.path.join(data_dir, filenames['MKSS'] + '_scanner.csv')

    imagedata, affine = imf.load_image(img_path)
    inv2mri_mat = imf.inv2mri(imagedata, affine)

    data = imf.load_mkss(mkss_path)
    coord_list = data.iloc[:, 16:19].values
    orient_list = data.iloc[:, 19:].values
    colour_list = data.iloc[:, 6:9].values
    size_list = data.iloc[:, 9].values
    label_list = data.iloc[:, 10].values.tolist()
    seed_list = data.iloc[:, 11:14].values
    # coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)

    seed_list_w = np.hstack((seed_list, np.ones([seed_list.shape[0], 1]))).transpose()
    seed_list_w = inv2mri_mat @ seed_list_w
    seed_list_w = seed_list_w.transpose()[:, :3]

    id_fids = ['LEI', 'REI', 'NAI']
    index_coord = label_list.index('BROCA')
    index_fids = [label_list.index(n) for n in id_fids]

    if TRANFORM_SCANNER:
        data_out = data[columns_export].copy()
        data_out.iloc[:, 1:4] = seed_list_w
        data_out['distance'] = -1 * np.ones([seed_list_w.shape[0], 1])
        data_out['distance_coil'] = -1 * np.ones([seed_list_w.shape[0], 1])
        for n in target_names:
            index_coord = label_list.index(n)
            distances = np.linalg.norm(seed_list_w - seed_list_w[index_coord], axis=1)
            distances_coil = np.linalg.norm(coord_list - coord_list[index_coord], axis=1)
            repeats_id = np.where((distances_coil != 0.0) & (distances_coil <= 5))
            target_id = np.where((distances_coil == 0.0))
            data_out.loc[repeats_id[0], 'label'] = n
            data_out.loc[target_id[0], 'is_target'] = True
            data_out.loc[repeats_id[0], 'distance'] = distances[repeats_id[0]]
            data_out.loc[repeats_id[0], 'distance_coil'] = distances_coil[repeats_id[0]]

    if SAVE_SCANNER:
        data_out.to_csv(save_path, sep=';', index=False, float_format="%.3f")

    distance = np.linalg.norm(coord_list[index_coord] - seed_list_w[index_coord])
    print("Distance between seed and coil center: {}".format(distance))

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

        # offset = 40
        # coil_norm = coil_norm/np.linalg.norm(coil_norm)
        # coord_offset = p1 - offset * coil_norm

        _ = vf.load_stl(coil_path, ren, opacity=.6, replace=repos_coil, colour=[1., 1., 1.], user_matrix=m_coil)

        _ = vf.add_line(ren, p1, p2_dir, color=[1.0, .0, .0])
        _ = vf.add_line(ren, p1, p2_face, color=[.0, 1.0, .0])
        _ = vf.add_line(ren, p1, p2_norm, color=[.0, .0, 1.0])

        _ = vf.add_marker(p1, ren, colours[4], radius=2)
        # --- coil vectors

        # seed markers
        _ = vf.add_marker(seed_list_w[index_coord], ren, colours[5], radius=2)
        # _ = vf.add_marker(seed_list[index_coord], ren, colours[6], radius=2)
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
