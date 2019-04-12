#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xcoord - Tools for cross-software spatial coordinate manipulation
#
# This file is part of xcoord package which is released under copyright.
# See file LICENSE or go to website for full license details.
# Copyright (C) 2018 Victor Hugo Souza - All Rights Reserved
#
# Homepage: https://github.com/vhosouza/xcoord
# Contact: victor.souza@aalto.fi
# License: MIT License
#
# Authors: Victor Hugo Souza, Renan Matsuda
# Date/version: 10.4.2019

import numpy as np
from nibabel.affines import apply_affine


def coord_change(img_shape, coord):

    reorder = [0, 2, 1]
    coord = [coord[s] for s in reorder]
    data_flip = coord.copy()
    data_flip[0] = img_shape[0] - coord[0]
    # data_flip[1] = img_shape[1] - coord[1]
    # data_flip[1] = - coord[1]

    return data_flip


def apply_affine2(affine, coord):

    # print(np.dot(affine, np.transpose(coord)))
    # coord.append(1.)
    coord = np.asarray(coord)[np.newaxis, :]

    # coord_transf = np.dot(affine[:3, :3], coord) + affine[:3, 3]
    # coord_transf = affine[:3, :3]@np.transpose(coord) + affine[:3, 3]

    # np.fill_diagonal(affine, 1.)
    # coord_transf = affine @ coord
    coord_transf = apply_affine(affine, coord)

    print(affine)

    return coord_transf[0, :].tolist()
    # return coord_transf.reshape([1, 4])[0, :-1].tolist()
