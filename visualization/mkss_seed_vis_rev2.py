#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

import image_funcs as imf

# %%
TRANFORM_SCANNER = True
SAVE_SCANNER = True

if sys.platform == "win32":
    onedrive_path = os.environ.get('OneDrive')
elif (sys.platform == "darwin") or (sys.platform == "linux"):
    onedrive_path = os.path.expanduser('~/OneDrive - Aalto University')
else:
    onedrive_path = False
    print("Unsupported platform")

marker_file = {'joonas': '20211123-200911-T1-markers_all_repeated',
               'pantelis': '20211126-121334-T1-markers_all_repeated',
               'baran': '20211123-205329-T1-markers_all_repeated',
               'victor': '20211126-132408-T1-markers_all_repeated'}

target_names = ['M1', 'BROCA', 'DLPFC', 'V1']
columns_export = ['label', 'x_seed', 'y_seed', 'z_seed', 'alpha', 'beta', 'gamma', 'is_target']

root_dir = os.path.join(onedrive_path, 'data', 'dti_navigation', 'normMRI')
save_path = os.path.join(root_dir, '20241124_stats_summary_rev2.csv')

columns_stats = ['subject', 'target', 'distance_seed', 'std_seed', 'distance_coil', 'std_coil' 'distance_orient', 'std_orient']
data_all = pd.DataFrame()

# subject_id = 0
# subject_list = ['joonas', 'pantelis', 'baran', 'victor']
# subject = subject_list[subject_id]

for subject in ['joonas', 'pantelis', 'baran', 'victor']:


    data_dir = os.path.join(root_dir, subject)

    filenames = {'MKSS': marker_file[subject],
                 'COIL': 'magstim_fig8_coil', 'T1': 'T1',
                 'BRAINSIM': 'wm', 'HEADSIM': 'skin'}

    # mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    mkss_path = os.path.join(data_dir, filenames['MKSS'] + '.mkss')
    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    # save_path = os.path.join(data_dir, filenames['MKSS'] + '_scanner.csv')

    imagedata, affine = imf.load_image(img_path)
    inv2mri_mat = imf.inv2mri(imagedata, affine)

    data = imf.load_mkss(mkss_path)
    coord_list = data.iloc[:, 16:19].values
    # orient_list = data.iloc[:, 19:].values
    # using the orientations in inv space because in world the errors for V1 are quite big
    # these are potentially caused by transformations with euler angles, but needs to be further
    # verified
    orient_list = data.iloc[:, 3:6].values
    colour_list = data.iloc[:, 6:9].values
    size_list = data.iloc[:, 9].values
    label_list = data.iloc[:, 10].values.tolist()
    seed_list = data.iloc[:, 11:14].values
    # coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)

    seed_list_w = np.hstack((seed_list, np.ones([seed_list.shape[0], 1]))).transpose()
    seed_list_w = inv2mri_mat @ seed_list_w
    seed_list_w = seed_list_w.transpose()[:, :3]

    # %
    id_fids = ['LEI', 'REI', 'NAI']
    index_coord = label_list.index('BROCA')
    index_fids = [label_list.index(n) for n in id_fids]

    if TRANFORM_SCANNER:
        data_out = data[columns_export].copy()
        data_out.iloc[:, 1:4] = seed_list_w
        data_out['distance'] = -1 * np.ones([seed_list_w.shape[0], 1])
        data_out['distance_coil'] = -1 * np.ones([seed_list_w.shape[0], 1])
        data_out['distance_orient'] = -1 * np.ones([seed_list_w.shape[0], 1])
        for n in target_names:
            index_coord = label_list.index(n)
            distances = np.linalg.norm(seed_list_w - seed_list_w[index_coord], axis=1)
            distances_coil = np.linalg.norm(coord_list - coord_list[index_coord], axis=1)
            distances_orient = np.linalg.norm(orient_list - orient_list[index_coord], axis=1)
            repeats_id = np.where((distances_coil != 0.0) & (distances_coil <= 5))
            target_id = np.where((distances_coil == 0.0))
            data_out.loc[repeats_id[0], 'label'] = n
            data_out.loc[target_id[0], 'is_target'] = True
            data_out.loc[repeats_id[0], 'distance'] = distances[repeats_id[0]]
            data_out.loc[repeats_id[0], 'distance_coil'] = distances_coil[repeats_id[0]]
            data_out.loc[repeats_id[0], 'distance_orient'] = distances_orient[repeats_id[0]]

    # %
    # Step 1: Select rows where a specific column (e.g., 'is_target') matches a value (e.g., True)
    selected_rows = data_out[data_out['distance'] > -1]
    # Step 2: Group data by another column (e.g., 'label')
    grouped_data = selected_rows.groupby('label')
    # Step 3: Print descriptive statistics for a numerical column (e.g., 'distance') based on the grouped data
    distance_seed_stats = grouped_data['distance'].describe()[['mean', 'std']]
    distance_coil_stats = grouped_data['distance_coil'].describe()[['mean', 'std']]
    distance_orient_stats = grouped_data['distance_orient'].describe()[['mean', 'std']]

    # Combine the results for distance and orientation
    all_stats = pd.concat([distance_seed_stats, distance_coil_stats, distance_orient_stats], axis=1,
                                      keys=['Seed', 'Coil', 'Orientation'])

    # Flatten the multi-level columns
    all_stats.columns = ['_'.join(col).strip() for col in all_stats.columns.values]
    all_stats = all_stats.reset_index()

    # Add the subject column
    all_stats['subject'] = subject

    # Reorder columns to place 'subject' column first
    cols = ['subject'] + [col for col in all_stats.columns if col != 'subject']
    all_stats = all_stats[cols]

    # Concatenate with the overall DataFrame
    data_all = pd.concat([data_all, all_stats], ignore_index=True)

# %%
if SAVE_SCANNER:
    data_all.to_csv(save_path, sep=';', index=False, float_format="%.3f")

# distance = np.linalg.norm(coord_list[index_coord] - seed_list_w[index_coord])
# print("Distance between seed and coil center: {}".format(distance))
#
# # create coil vectors
# coil_pos = np.hstack((coord_list[index_coord], orient_list[index_coord]))
# m_coil = imf.coil_transform_matrix(coil_pos)
#
# vec_length = 75
# repos_coil = [0., 0., 0., 0., 0., 90.]
#
# # coil vectors in invesalius 3D space
# p1 = m_coil[:-1, -1]
# coil_dir = m_coil[:-1, 0]
# coil_face = m_coil[:-1, 1]
# p2_face = p1 + vec_length * coil_face
# p2_dir = p1 + vec_length * coil_dir
# coil_norm = np.cross(coil_dir, coil_face)
# p2_norm = p1 - vec_length * coil_norm


