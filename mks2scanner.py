#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import image_funcs as imf


def main():
    FIX_MKS = False
    SAVE_FIX = False
    SAVE_SCAN = True
    RECOMPUTE_SEED = False

    SEED_OFFSET = 25

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'

    filenames = {'T1': 'sub-S1_ses-S8741_T1w', 'MKS': 'markers_20210304_m1',
                 'MKS_CORR': 'markers_20210304_m1', 'ACT': 'trekkerACTlabels'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    act_path = os.path.join(data_dir, filenames['ACT'] + '.nii')
    mkss_path = os.path.join(data_dir, filenames['MKS'] + '.mks')
    mkss_save_path = os.path.join(data_dir, filenames['MKS'] + '_scanner.mkss')

    if FIX_MKS:
        mks_corrupt_path = os.path.join(data_dir, filenames['MKS_CORR'] + '.mks')
        content = imf.load_mks_corrupt(mks_corrupt_path)
        content_fixed = imf.fix_mks_corrupt(content, problems=['0.5019607843137255', '1.0'])

        if SAVE_FIX:
            mkss_fix_path = os.path.join(data_dir, filenames['MKS_CORR'] + '_fix.mkss')
            imf.save_mks(mkss_fix_path, content_fixed)

    else:
        imagedata, affine = imf.load_image(img_path)
        # mri2inv_mat = imf.mri2inv(imagedata, affine)
        inv2mri_mat = imf.inv2mri(imagedata, affine)
        recomp_seed = {'recompute': False, 'recompute': False, 'seed_offset': 25, 'act_data': None, 'affine': None}

        if RECOMPUTE_SEED:
            imagedata_act, _ = imf.load_image(act_path)
            act_data = imagedata_act.get_fdata()
            recomp_seed = {'recompute': RECOMPUTE_SEED, 'seed_offset': SEED_OFFSET, 'act_data': act_data,
                           'affine': affine}

        coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list = imf.load_mks(mkss_path)
        coil_pos_scan, seed_scan = imf.mks2mkss(coord_list, orient_list, seed_list,
                                                user_matrix=inv2mri_mat, recompute_seed=recomp_seed)

        if SAVE_SCAN:
            # content_scan = coil_pos_scan.tolist(), colour_list, size_list, id_list, seed_scan.tolist(), tg_list
            content_scan = coil_pos_scan, colour_list, size_list, id_list, seed_scan, tg_list
            imf.save_mkss_scan(filename=mkss_save_path, content=content_scan)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.set_printoptions(suppress=True, precision=2)
    main()
