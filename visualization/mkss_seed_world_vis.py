#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import image_funcs as imf
from visualization import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True
    SHOW_COIL = True
    SHOW_FIDUCIALS = False

    subject_id = 0
    subject = ['victor']
    marker_file = {'victor': '20211128-185400-T1-markers'}

    root_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\normMRI'
    data_dir = os.path.join(root_dir, subject[subject_id])

    filenames = {'MKSS': marker_file[subject[subject_id]],
                 'COIL': 'magstim_fig8_coil', 'T1': 'T1',
                 'BRAINSIM': 'wm', 'HEADSIM': 'skin'}

    # mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    head_sim_path = os.path.join(data_dir, filenames['HEADSIM'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    coil_path = os.path.join(root_dir, filenames['COIL'] + '.stl')

    data = imf.load_mkss(mkss_path)
    coord_list = data.iloc[:, 16:19].values
    orient_list = data.iloc[:, 19:].values
    colour_list = data.iloc[:, 6:9].values
    size_list = data.iloc[:, 9].values
    label_list = data.iloc[:, 10].values.tolist()
    seed_list = data.iloc[:, 11:14].values
    # coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)

    if SHOW_FIDUCIALS:
        id_fids = ['LEI', 'REI', 'NAI']
        index_fids = [label_list.index(n) for n in id_fids]

    index_coord = label_list.index('*')

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
        if SHOW_FIDUCIALS:
            for n in index_fids:
                _ = vf.add_marker(coord_list[n], ren, colours[n], radius=2)
        # --- fiducial markers

        # create coil vectors
        if SHOW_COIL:
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
        _ = vf.add_marker(seed_list[index_coord], ren, colours[5], radius=2)
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
