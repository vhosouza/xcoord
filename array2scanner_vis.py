#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import csv
import numpy as np

from visualization import vis_funcs as vf


def main():
    """
Load the stimulus locations (markers) collected in the motor mapping experiment (Tardelli et al., 2021) with
InVesalius and visualize relative to the cortical surface. All coordinates are already in the scanner space.
Victor Souza 11.10.2021
    """
    SHOW_WINDOW = True
    SHOW_AXES = False
    SAVE_IMAGE = True

    intensity = 120  # [110, 120]
    muscles = 'FRC'  # ['FCP', 'FRC', 'ADM']
    subject = 11  # [4, 5, 7, 9]

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\motor_mapping'

    subject_file = '{:02d}_{}_MRI_{}_processed_scanner'.format(subject, intensity, muscles)

    filenames = {'T1': 'T1_sub-{:02d}'.format(subject), 'CSV': subject_file,
                 'BRAINSIM': 'gm_sn_sub-{}'.format(subject), 'EXPORT_PNG': subject_file}

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
    coord_avg = np.vstack([coord_list[:, n].reshape(-1, 3).mean(axis=1) for n in range(3)]).T

    if SHOW_WINDOW:
        # Create a rendering window and renderer
        camera_cfg = {'azimuth': 110, 'elevation': 50, 'focal_point': 3 * [0], 'position': (0, 750, 0)}
        ren, ren_win, iren = vf.create_window(background=(1., 1., 1.), camera_cfg=camera_cfg)

        repos = [0., 0., 0., 0., 0., 0.]

        _ = vf.load_stl(brain_sim_path, ren, opacity=1., colour="SkinColor", replace=repos,
                     user_matrix=np.identity(4))

        for coord in coord_avg:
            _ = vf.add_marker(coord, ren, [1., 0., 0.], radius=2)

        # Add axes to scene origin
        if SHOW_AXES:
            _ = vf.add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
            _ = vf.add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
            _ = vf.add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

        # Export window to PNG, initialize window and interactor

        ren.ResetCameraClippingRange()
        ren_win.Render()

        if SAVE_IMAGE:
            filename_png = os.path.join(data_dir, filenames['EXPORT_PNG'] + '.png')
            vf.export_window_png(filename_png, ren_win)

        iren.Initialize()
        iren.Start()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.set_printoptions(suppress=True, precision=2)
    main()
