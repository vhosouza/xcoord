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

# Transformation to correct place required a surface with no coordinate change from simnibs
# Transformations from nexstim to mri space is: swap y and z coords, flip x coord, then apply affine from image header


def coord_change(coord, img_shape, affine=np.identity(4), flipxyz=[False, False, False], axis_order=[0, 1, 2]):
    flipx, flipy, flipz = flipxyz

    # swap axis
    coord = [coord[s] for s in axis_order]
    data_flip = coord.copy()

    # flip axis
    if flipx:
        data_flip[0] = img_shape[0] - coord[0]
    if flipy:
        data_flip[1] = img_shape[1] - coord[1]
    if flipz:
        data_flip[2] = img_shape[2] - coord[2]

    # apply the affine matrix from nifti image header
    # this converts from mri to world (scanner) space
    # https://nipy.org/nibabel/coordinate_systems.html#the-affine-matrix-as-a-transformation-between-spaces
    M = affine[:3, :3]
    abc = affine[:3, 3]
    coord_transf = M.dot(data_flip) + abc

    return coord_transf.tolist()
