#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import csv
import numpy as np

import image_funcs as imf
import vis_funcs as vf


def main():
    SHOW_WINDOW = True
    SHOW_AXES = True

    intensity = 110  # [110, 120]
    muscles = 'FCP'  # ['FCP', 'FRC', 'ADM']
    subject = 4  # [4, 5, 7, 9]

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\motor_mapping'

    filenames = {'T1': 'T1_sub-{:02d}'.format(subject),
                 'CSV': '{:02d}_{}_MRI_{}_processed_scanner'.format(subject, intensity, muscles),
                 'BRAINSIM': 'gm_sn'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    csv_path = os.path.join(data_dir, 'csv', filenames['CSV'] + '.csv')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')

    data = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        for row in reader:
            data.append([float(s) for s in row])

    data_arr = np.asarray(data)
    coord_list = data_arr[:, :3].copy()

    # imagedata, affine = imf.load_image(img_path)
    # mri2inv_mat = imf.mri2inv(imagedata, affine)
    # inv2mri_mat = imf.inv2mri(imagedata, affine)

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        ren, ren_win, iren = vf.create_window()

        # 0: red, 1: green, 2: blue, 3: maroon (dark red),
        # 4: purple, 5: teal (petrol blue), 6: yellow, 7: orange
        colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
                   [0.45, 0., 0.5], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

        repos = [0., 0., 0., 0., 0., 0.]

        _ = vf.load_stl(brain_sim_path, ren, opacity=.6, colour=[1., 1., 1.], replace=repos,
                     user_matrix=np.identity(4))

        for coord in coord_list:
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
