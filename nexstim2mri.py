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


def coord_change(img_shape, coord):

    reorder = [0, 2, 1]
    coord = [coord[s] for s in reorder]
    data_flip = coord.copy()
    data_flip[0] = img_shape[0] - coord[0]
    # data_flip[1] = img_shape[1] - coord[1]

    return data_flip
