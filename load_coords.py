#!/usr/bin/env python3
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
# Authors: Victor Hugo Souza
# Date/version: 10.4.2019

import re
from itertools import zip_longest


def load_nexstim(filename):
    
    with open(filename, 'r') as f:
        x = f.readlines()

    x = [e for e in x if e != '\n']
    x = [e.replace('\n', '') for e in x]
    x = [e[1:] if e[0] == '\t' else e for e in x]

    pattern = 'Landmarks'
    for n, s in enumerate(x):
        seq_found = re.search(pattern, s)
        if seq_found:
            final = [z.split('\t') for z in x[n + 1:n + 8]]
            fids = [t for t in zip_longest(*final)]
            break

    reorder = [3, 0, 1, 2]
    fids = [fids[s] for s in reorder]

    pattern = 'Time'
    for n, s in enumerate(x):
        seq_found = re.search(pattern, s)
        if seq_found:
            final = [z.split('\t') for z in x[n:]]
            stim_info = [t for t in zip_longest(*final)]
            break

    rel_data = [[] for s in range(len(fids[0]))]

    for s in fids:
        for n, txt in enumerate(s):
            rel_data[n].append(txt)

    for n, txt in enumerate(stim_info):
        if txt[0] == 'Coil' or (txt[0] == 'EF max.' and txt[1] == 'Loc.'):
            rel_data.append([txt[0] + ' ' + txt[1], stim_info[n][-1], stim_info[n + 1][-1], stim_info[n + 2][-1]])

    data = [[] for s in range(len(rel_data))]
    for n, txt in enumerate(rel_data):
        data[n] = [c if n == 0 or k == 0 else float(c) for k, c in enumerate(txt)]

    return data

