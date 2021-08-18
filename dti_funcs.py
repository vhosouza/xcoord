#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Trekker
import vtk
import numpy as np
import time
import psutil


def start_trekker(filename, params):

    trekker = Trekker.tracker(filename.encode('utf-8'))
    trekker.seed_maxTrials(params['seed_max'])
    trekker.stepSize(params['step_size'])
    trekker.minFODamp(params['min_fod'])
    trekker.probeQuality(params['probe_quality'])
    trekker.maxEstInterval(params['max_interval'])
    trekker.minRadiusOfCurvature(params['min_radius_curv'])
    trekker.probeLength(params['probe_length'])
    trekker.writeInterval(params['write_interval'])
    trekker.numberOfThreads(params['numb_threads'])

    return trekker


def simple_direction(trk_n, alpha=255):
    # trk_d = np.diff(trk_n, axis=0, append=2*trk_n[np.newaxis, -1, :])
    trk_d = np.diff(trk_n, axis=0, append=trk_n[np.newaxis, -2, :])
    trk_d[-1, :] *= -1
    # check that linalg norm makes second norm
    # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    direction = 255 * np.absolute((trk_d / np.linalg.norm(trk_d, axis=1)[:, None]))
    direction = np.hstack([direction, alpha*np.ones([direction.shape[0], 1])])
    return direction.astype(int)
    # return direction


def compute_tubes_vtk(trk, direction):
    numb_points = trk.shape[0]
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)

    k = 0
    lines.InsertNextCell(numb_points)
    for j in range(numb_points):
        points.InsertNextPoint(trk[j, :])
        colors.InsertNextTuple(direction[j, :])
        lines.InsertCellPoint(k)
        k += 1

    trk_data = vtk.vtkPolyData()
    trk_data.SetPoints(points)
    trk_data.SetLines(lines)
    trk_data.GetPointData().SetScalars(colors)

    # make it a tube
    trk_tube = vtk.vtkTubeFilter()
    trk_tube.SetRadius(0.5)
    trk_tube.SetNumberOfSides(4)
    trk_tube.SetInputData(trk_data)
    trk_tube.Update()

    return trk_tube


def tracts_root(out_list):
    branch = vtk.vtkMultiBlockDataSet()
    # create tracts only when at least one was computed
    # print("Len outlist in root: ", len(out_list))
    if not out_list.count(None) == len(out_list):
        for n, tube in enumerate(out_list):
            #TODO: substitute to try + except (better to ask forgiveness than please)
            # if tube:
            branch.SetBlock(n, tube.GetOutput())

    return branch


def tracts_computation(trk_list, root, n_tracts):
    # Transform tracts to array
    start_time = time.time()
    trk_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in trk_list]
    duration = time.time() - start_time
    print("Tracts to array: {:.2f} ms".format(1e3*duration))

    # Compute the directions
    start_time = time.time()
    trk_dir = [simple_direction(trk_n) for trk_n in trk_arr]
    duration = time.time() - start_time
    print("Tracts directions: {:.2f} ms".format(1e3*duration))

    # Compute the vtk tubes
    start_time = time.time()
    out_list = [compute_tubes_vtk(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(trk_arr, trk_dir)]
    duration = time.time() - start_time
    print("Compute tubes: {:.2f} ms".format(1e3*duration))

    start_time = time.time()
    root = tracts_root(out_list)
    duration = time.time() - start_time
    print("Compute root: {:.2f} ms".format(1e3*duration))

    return root


def tracts_computation_noroot(trk_list, alpha=255):
    # Transform tracts to array
    trk_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in trk_list]
    # Compute the directions
    trk_dir = [simple_direction(trk_n, alpha=alpha) for trk_n in trk_arr]
    # Compute the vtk tubes
    out_list = [compute_tubes_vtk(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(trk_arr, trk_dir)]

    return out_list


def compute_actor(root, affine_vtk):
    mapper = vtk.vtkCompositePolyDataMapper2()
    mapper.SetInputDataObject(root)

    actor_tracts = vtk.vtkActor()
    actor_tracts.SetMapper(mapper)
    actor_tracts.SetUserMatrix(affine_vtk)

    return actor_tracts


def single_block(tracker, seed, n_tracts, root, matrix_vtk):
    start_time = time.time()
    tracker.seed_coordinates(np.repeat(seed, n_tracts, axis=0))
    duration = time.time() - start_time
    print("Seed coordinates: {:.2f} ms".format(1e3*duration))

    start_time = time.time()
    trk_list = tracker.run()
    duration = time.time() - start_time
    print("Run Trekker: {:.2f} ms".format(1e3*duration))

    start_time = time.time()
    root = tracts_computation(trk_list, root, 0)
    duration = time.time() - start_time
    print("Tracts computation: {:.2f} ms".format(1e3*duration))
    # visualizeTracks(ren, ren_win, tracker, seed, user_matrix=matrix_vtk)

    start_time = time.time()
    tracts_actor = compute_actor(root, matrix_vtk)
    duration = time.time() - start_time
    print("Compute actor: {:.2f} ms".format(1e3*duration))

    return tracts_actor


def multi_block(tracker, seed, n_threads):
    tracker.seed_coordinates(np.repeat(seed, n_threads, axis=0))
    trk_list = tracker.run()
    out_list = tracts_computation_noroot(trk_list)
    branch = tracts_root(out_list)

    return branch


def multi_block_uncertainty(trekker, seed, n_threads, n):
    trekker.seed_coordinates(np.repeat(seed, n_threads, axis=0))
    alpha = (n - 1) * (255 - 51) / (10 - 1) + 51
    trekker.minFODamp(n*0.01)
    trekker.dataSupportExponent(n*0.1)
    trk_list = trekker.run()
    out_list = tracts_computation_noroot(trk_list, alpha)
    branch = tracts_root(out_list)

    return branch


def set_trekker_parameters(trekker, params):
    """Set all user-defined parameters for tractography computation using the Trekker library

    :param trekker: Trekker instance
    :type trekker: Trekker.T
    :param params: Dictionary containing the parameters values to set in Trekker. Initial values are in constants.py
    :type params: dict
    :return: List containing the Trekker instance and number of threads for parallel processing in the computer
    :rtype: list
    """
    trekker.seed_maxTrials(params['seed_max'])
    # trekker.stepSize(params['step_size'])
    trekker.minFODamp(params['min_fod'])
    # trekker.probeQuality(params['probe_quality'])
    # trekker.maxEstInterval(params['max_interval'])
    # trekker.minRadiusOfCurvature(params['min_radius_curv'])
    # trekker.probeLength(params['probe_length'])
    trekker.writeInterval(params['write_interval'])
    trekker.maxLength(params['max_lenth'])
    trekker.minLength(params['min_lenth'])
    trekker.maxSamplingPerStep(params['max_sampling_step'])

    # check number if number of cores is valid in configuration file,
    # otherwise use the maximum number of threads which is usually 2*N_CPUS
    n_threads = 2 * psutil.cpu_count()
    if isinstance((params['numb_threads']), int) and params['numb_threads'] <= 2*const.N_CPU:
        n_threads = params['numb_threads']

    trekker.numberOfThreads(n_threads)
    # print("Trekker config updated: n_threads, {}; seed_max, {}".format(n_threads, params['seed_max']))
    return trekker, n_threads