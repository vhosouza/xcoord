#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import csv
import numpy as np
import pandas as pd

import image_funcs as imf


def main():
    SAVE_SCAN = True

    intensity = [110, 120]
    muscles = ['FCP', 'FRC', 'ADM']
    subject = [4, 5, 7, 9]

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\motor_mapping'

    for subj_ in subject:
        for int_ in intensity:
            for musc_ in muscles:

                # filenames = {'T1': 'subject04_victor', 'XLS': '04_110_MRI_ADM_processed',
                #              'XLS_CORR': '04_110_MRI_ADM_processed_scanner'}

                filenames = {'T1': 'T1_sub-{:02d}'.format(subj_), 'XLS': '{:02d}_{}_MRI_{}_processed'.format(subj_, int_, musc_)}

                img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
                xls_path = os.path.join(data_dir, filenames['XLS'] + '.xlsx')
                csv_save_path = os.path.join(data_dir, filenames['XLS'] + '_scanner.csv')

                imagedata, affine = imf.load_image(img_path)
                # mri2inv_mat = imf.mri2inv(imagedata, affine)
                inv2mri_mat = imf.inv2mri(imagedata, affine)

                header = ['x', 'y', 'z', 'ch1_amp', 'ch1_lat', 'ch2_amp', 'ch2_lat', 'ch3_amp', 'ch3_lat']
                data = pd.read_excel(xls_path, header=None,
                                   names=header, usecols='D:F,I:N')
                data.drop(labels=range(6), axis=0, inplace=True)
                data_arr = data.to_numpy()
                data_arr = data_arr.astype(float)

                coord_list = data_arr[:, :3].copy()
                orient_list = np.zeros([data_arr.shape[0], 3])

                coil_pos_scan = imf.array2scanner(coord_list, orient_list, user_matrix=inv2mri_mat)

                data_scanner = data_arr.copy()
                data_scanner[:, :3] = coil_pos_scan[:, :3]

                if SAVE_SCAN:
                    with open(csv_save_path, 'w', newline='') as f:
                        writer = csv.writer(f, delimiter=';')
                        # write the header
                        writer.writerow(header)
                        # write multiple rows
                        writer.writerows(data_scanner)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.set_printoptions(suppress=True, precision=2)
    main()
