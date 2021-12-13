#!/usr/bin/env python
# -*- coding: utf-8 -*-
import vtk
import numpy as np
import csv
import external.transformations as tf
import nibabel as nb
import pandas as pd
import scipy.io as sio


def load_image(filename, closest=True):
    imagedata = nb.squeeze_image(nb.load(filename))
    if closest:
        imagedata = nb.as_closest_canonical(imagedata)
        imagedata.update_header()
    # pix_dim = imagedata.header.get_zooms()
    # img_shape = imagedata.header.get_data_shape()

    # print("Pixel size: \n")
    # print(pix_dim)
    # print("\nImage shape: \n")
    # print(img_shape)
    #
    # print("\nSform: \n")
    # print(imagedata.get_qform(coded=True))
    # print("\nQform: \n")
    # print(imagedata.get_sform(coded=True))
    # print("\nFall-back: \n")
    # print(imagedata.header.get_base_affine())

    scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
    affine_noscale = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

    # print("Affine no scale: {0}\n".format(affine))
    return imagedata, affine_noscale


def mks2mkss(coord_list, orient_list, seed_list, user_matrix=None,
             recompute_seed={'recompute': False, 'seed_offset': 25, 'act_data': None, 'affine': None}):

    n_points = coord_list.shape[0]
    recompute = recompute_seed['recompute']
    seed_offset = recompute_seed['seed_offset']
    act_data = recompute_seed['act_data']
    affine = recompute_seed['affine']

    coord_w = np.hstack((coord_list, np.ones((n_points, 1))))
    seed_w = np.hstack((seed_list, np.ones((n_points, 1))))

    if recompute:
        coord_list_w = create_grid((-2, 2), (0, 20), seed_offset - 5, 1)

    # coil vectors in MRI space
    coil_pos = np.hstack((coord_w[:, :3], orient_list[:, :]))
    coil_pos[:, 1] *= -1
    coil_pos_scan = np.empty_like(coil_pos)
    seed_scan = np.empty_like(seed_list)
    for n in range(coil_pos.shape[0]):
        m_coil = coil_transform_matrix(coil_pos[n, :])
        m_coil_scan = user_matrix @ m_coil
        coil_pos_scan[n, :] = coil_transform_pos(m_coil_scan)

        if recompute:
            coord_list_w_tr = m_coil_scan @ coord_list_w
            seed_scan_aux = grid_offset(act_data, coord_list_w_tr, affine=affine)
            if seed_scan_aux is not None:
                seed_scan[n, :] = seed_scan_aux

    # seed markers
    if not recompute:
        seed_3dinv = seed_w.copy()
        seed_scan = user_matrix @ seed_3dinv.T
        seed_scan = seed_scan.T[:, :3]
    # --- seed markers

    return coil_pos_scan.tolist(), seed_scan.tolist()


def array2scanner(coord_list, orient_list, user_matrix=None):

    n_points = coord_list.shape[0]

    coord_w = np.hstack((coord_list, np.ones((n_points, 1))))

    # coil vectors in MRI space
    coil_pos = np.hstack((coord_w[:, :3], orient_list[:, :]))
    coil_pos[:, 1] *= -1
    coil_pos_scan = np.empty_like(coil_pos)
    for n in range(coil_pos.shape[0]):
        m_coil = coil_transform_matrix(coil_pos[n, :])
        m_coil_scan = user_matrix @ m_coil
        coil_pos_scan[n, :] = coil_transform_pos(m_coil_scan)

    return coil_pos_scan


def mri2inv(imagedata, affine=None):
    """
    Affine transformation from world to InVesalius space.

    The affine from image header converts from the voxel to the scanner space. Thus, the inverse of the affine converts
    from the scanner to the voxel space.
    InVesalius and voxel spaces are otherwise identical, but InVesalius space has a reverted y-axis
    (increasing y-coordinate moves posterior in InVesalius space, but anterior in the voxel space).

    For instance, if the size of the voxel image is 256 x 256 x 160, the y-coordinate 0 in
    InVesalius space corresponds to the y-coordinate 255 in the voxel space.

    :param position: a vector of 3 coordinates (x, y, z) in InVesalius space.
    :return: a vector of 3 coordinates in the voxel space
    """
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    if affine is None:
        scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
        affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

    # convert from invesalius 3D to RAS+ space
    mri2inv_mat = np.linalg.inv(affine.copy())
    mri2inv_mat[1, 3] = mri2inv_mat[1, -1] - pix_dim[1]*(img_shape[1] - 1)

    return mri2inv_mat


def inv2mri(imagedata, affine=None):
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    if affine is None:
        scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
        affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

    mri2inv = np.linalg.inv(affine.copy())
    mri2inv[1, -1] = mri2inv[1, -1] - pix_dim[1]*(img_shape[1] - 1)
    inv2mri_mat = np.linalg.inv(mri2inv)

    return inv2mri_mat


def load_mks(filename):
    BTNS_IMG_MKS = {'IR1': {0: 'LEI'},
                    'IR2': {1: 'REI'},
                    'IR3': {2: 'NAI'}}
    count_line = 0
    list_coord = []

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        if filename.lower().endswith('.mkss'):
            # skip the header
            next(reader)
        content = [row for row in reader]

    for line in content:
        target = None
        if len(line) > 8:
            coord = [float(s) for s in line[:6]]
            colour = [float(s) for s in line[6:9]]
            size = float(line[9])
            marker_id = line[10]

            if len(line) > 11:
                seed = [float(s) for s in line[11:14]]
            else:
                seed = 3 * [0.]

            if len(line) == 15:
                target_id = line[14]
            else:
                target_id = '*'

            if len(line) >= 11:
                for i in BTNS_IMG_MKS:
                    if marker_id in list(BTNS_IMG_MKS[i].values())[0]:
                        print("Found fiducial {}: {}".format(marker_id, coord))
                        # Publisher.sendMessage('Load image fiducials', marker_id=marker_id, coord=coord)
                    elif marker_id == 'TARGET':
                        target = count_line
            else:
                marker_id = '*'

            # data = [coord, colour, size, marker_id, seed, target_id]

            # if there are multiple TARGETS will set the last one
            # if target:
            #     self.OnMenuSetTarget(target)

        else:
            # for compatibility with previous version without the extra seed and target columns
            coord = float(line[0]), float(line[1]), float(line[2]), 0, 0, 0
            colour = float(line[3]), float(line[4]), float(line[5])
            size = float(line[6])
            seed = 3 * [0]
            target_id = '*'

            if len(line) == 8:
                marker_id = line[7]
                for i in BTNS_IMG_MKS:
                    if marker_id in list(BTNS_IMG_MKS[i].values())[0]:
                        print("Found fiducial {}: {}".format(marker_id, coord))
                        # Publisher.sendMessage('Load image fiducials', marker_id=marker_id, coord=coord)
            else:
                marker_id = '*'

        # data = coord, colour, size, marker_id, seed, target_id

        # List marker properties
        line = 15 * [0]
        line[:6] = [round(s, 1) for s in coord]
        line[6:9] = [round(s, 3) for s in colour]
        line[9] = round(size, 1)
        line[10] = marker_id
        line[11:14] = [round(s, 1) for s in seed]
        line[14] = target_id

        list_coord.append(line)

        count_line += 1

    # convert from lists of lists to actual separate lists for each data with proper types for transformations
    coord_list = np.array([s[:3] for s in list_coord])
    orient_list = np.array([s[3:6] for s in list_coord])
    colour_list = [s[6:9] for s in list_coord]
    size_list = [s[9] for s in list_coord]
    id_list = [s[10] for s in list_coord]
    seed_list = np.array([s[11:14] for s in list_coord])
    tg_list = [s[14] for s in list_coord]

    return coord_list, orient_list, colour_list, size_list, id_list, seed_list, tg_list


def load_mks_corrupt(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        if filename.lower().endswith('.mkss'):
            # skip the header
            next(reader)
        content = [row for row in reader]

    return content


def load_mkss(filename):
    return pd.read_csv(filename, delimiter='\t', skiprows=1)


def fix_mks_corrupt(content, problems):
    list_coord = []
    for row in content.copy():
        if len(row) == 15 and problems:
            if problems[0] in row:
                row.remove(problems[0])
            elif problems[1]:
                row.remove(problems[1])

            mk_id = row[-5]
            if mk_id != '2':
                row[-4] = mk_id
                row[-5] = '2'
        if row[-4] == '':
            row[-4] = '*'

        list_coord.append(row)

    return list_coord


def save_mks(filename, content):
    data = []
    for row in content:
        line = 15 * [0]
        line[:6] = [round(float(s), 1) for s in row[:6]]
        line[6:9] = [round(float(s), 3) for s in row[6:9]]
        line[9] = round(float(row[9]), 1)

        if row[10] == 'x':
            line[10] = '*'
        else:
            line[10] = row[10]

        line[11:14] = [round(float(s), 1) for s in row[11:14]]

        if row[10] == 'TARGET':
            line[14] = 'TARGET'
        else:
            line[14] = '*'

        data.append(line)

    header_titles = ['x', 'y', 'z', 'alpha', 'beta', 'gamma', 'r', 'g', 'b',
                     'size', 'marker_id', 'x_seed', 'y_seed', 'z_seed', 'target_id']

    if filename and data:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            if filename.lower().endswith('.mkss'):
                writer.writerow(header_titles)
            writer.writerows(data)


def save_mkss_scan(filename, content):
    coord, colour, size, marker_id, seed, target_id = content
    data = []
    for n in range(len(marker_id)):
        line = 15 * [0]
        line[:6] = [round(s, 1) for s in coord[n]]
        line[6:9] = [round(s, 3) for s in colour[n]]
        line[9] = round(size[n], 1)
        line[10] = marker_id[n]
        line[11:14] = [round(s, 1) for s in seed[n]]
        line[14] = target_id[n]

        data.append(line)

    header_titles = ['x', 'y', 'z', 'alpha', 'beta', 'gamma', 'r', 'g', 'b',
                     'size', 'marker_id', 'x_seed', 'y_seed', 'z_seed', 'target_id']

    if filename and data:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(header_titles)
            writer.writerows(data)


def coil_transform_matrix(pos):
    a, b, g = np.radians(pos[3:])
    r_ref = tf.euler_matrix(a, b, g, 'sxyz')
    t_ref = tf.translation_matrix(pos[:3])
    m_img = tf.concatenate_matrices(t_ref, r_ref)

    return m_img


def coil_transform_pos(mcoil):
    _, _, angs, trans, _ = tf.decompose_matrix(mcoil)
    ang_deg = np.degrees(angs)

    return np.hstack((trans, ang_deg))


def load_eeg_mat(filename, variable='digitized_points'):
    mat = sio.loadmat(filename)
    if variable in mat:
        points = mat[variable]
    else:
        raise KeyError("Variable name not found")
        points = None

    return points


def create_grid(xy_range, z_range, z_offset, spacing):
    x = np.arange(xy_range[0], xy_range[1]+1, spacing)
    y = np.arange(xy_range[0], xy_range[1]+1, spacing)
    z = z_offset + np.arange(z_range[0], z_range[1]+1, spacing)
    xv, yv, zv = np.meshgrid(x, y, -z)
    coord_grid = np.array([xv, yv, zv])
    # create grid of points
    grid_number = x.shape[0]*y.shape[0]*z.shape[0]
    coord_grid = coord_grid.reshape([3, grid_number]).T
    # sort grid from distance to the origin/coil center
    coord_list = coord_grid[np.argsort(np.linalg.norm(coord_grid, axis=1)), :]
    # make the coordinates homogeneous
    coord_list_w = np.append(coord_list.T, np.ones([1, grid_number]), axis=0)

    return coord_list_w


def grid_offset(data, coord_list_w_tr, affine=np.identity(4)):
    # convert to int so coordinates can be used as indices in the MRI image space
    coord_list_w_tr_mri = np.linalg.inv(affine) @ coord_list_w_tr
    coord_list_w_tr_mri_int = coord_list_w_tr_mri[:3, :].T.astype(int)

    # extract the first occurrence of a specific label from the MRI image
    try:
        labs = data[coord_list_w_tr_mri_int[..., 0], coord_list_w_tr_mri_int[..., 1], coord_list_w_tr_mri_int[..., 2]]
        lab_first = np.where(labs == 1)
        if not lab_first:
            pt_found_w = None
        else:
            pt_found_w = coord_list_w_tr[:, lab_first[0][0]][:3]
    except IndexError:
        print("Warning: Index error when computing seed")
        pt_found_w = (1., 1., 1)

    return pt_found_w


def grid_offset_inv(data, coord_list_w_tr, img_shift):
    # convert to int so coordinates can be used as indices in the MRI image space
    coord_list_w_tr_mri = coord_list_w_tr[:3, :].T.astype(int) + np.array([[0, img_shift, 0]])

    #FIX: IndexError: index 269 is out of bounds for axis 2 with size 256
    # error occurs when running line "labs = data[coord..."
    # need to check why there is a coordinate outside the MRI bounds

    # extract the first occurrence of a specific label from the MRI image
    labs = data[coord_list_w_tr_mri[..., 0], coord_list_w_tr_mri[..., 1], coord_list_w_tr_mri[..., 2]]
    lab_first = np.argmax(labs == 1)
    if labs[lab_first] == 1:
        pt_found = coord_list_w_tr_mri[lab_first, :]
        # convert coordinate back to invesalius 3D space
        pt_found_inv = pt_found - np.array([0., img_shift, 0.])
    else:
        pt_found_inv = None

    # # convert to world coordinate space to use as seed for fiber tracking
    # pt_found_tr = np.append(pt_found, 1)[np.newaxis, :].T
    # # default affine in invesalius is actually the affine inverse
    # pt_found_tr = np.linalg.inv(affine) @ pt_found_tr
    # pt_found_tr = pt_found_tr[:3, 0, np.newaxis].T

    return pt_found_inv
