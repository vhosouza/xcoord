#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import pickle
import vtk

"""
Minimum working example that visualizes a set of computed tracts as vtk tubes.
(C) 2019 Victor Hugo Souza and Dogu Baran Aydogan
"""


def compute_direction(trk_n):
    # Compute the direction of each point in the tract and convert to an RGB list of colors from 0 to 255
    # Duplicate the last point to have the same color as the previous, as diff results in an n-1 array
    trk_d = np.diff(trk_n, axis=0, append=trk_n[np.newaxis, -2, :])
    trk_d[-1, :] *= -1
    direction = 255 * np.absolute((trk_d / np.linalg.norm(trk_d, axis=1)[:, None]))
    return direction.astype(int)


def compute_tubes(trk, direc):
    # trk: the array of coordinates that defines one tract
    # direc: the RGB color defined by the local direction of each coordinate

    numb_points = trk.shape[0]
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)

    lines.InsertNextCell(numb_points)

    for j in range(numb_points):
        points.InsertNextPoint(trk[j, :])
        lines.InsertCellPoint(j)
        colors.InsertNextTuple(direc[j, :])

    trk_poly = vtk.vtkPolyData()
    trk_poly.SetPoints(points)
    trk_poly.SetLines(lines)
    trk_poly.GetPointData().SetScalars(colors)

    # make it a tube
    # trk_tube = vtk.vtkTubeFilter()
    # trk_tube.SetRadius(0.5)
    # trk_tube.SetNumberOfSides(4)
    # trk_tube.SetInputData(trk_poly)
    # trk_tube.Update()

    return trk_poly


def visualize_tracts(out_list):
    # Create a single block dataset from all the tubes representing the tracts
    # One actor is easier to manipulate
    # Create tracts only when at least one was computed
    if not out_list.count(None) == len(out_list):
        root = vtk.vtkMultiBlockDataSet()

        for n, tube in enumerate(out_list):
            if tube:
                root.SetBlock(n, tube)

        # https://lorensen.github.io/VTKExamples/site/Python/CompositeData/CompositePolyDataMapper/
        mapper = vtk.vtkCompositePolyDataMapper2()
        mapper.SetInputDataObject(root)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

    return actor


if __name__ == "__main__":

    # Load list of tracts coordinates computed using the Trekker library
    with open('tracts_list.trk', 'rb') as fp:
        coords_list = pickle.load(fp)

    # Create a rendering window, renderer and interactor
    renderer = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetSize(640, 480)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(ren_win)

    # Convert the list of coordinates to numpy arrays
    start_time = time.time()
    coord_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in coords_list]
    duration = time.time() - start_time
    print("Convert to array: {:.2f} ms".format(1e3*duration))

    # Compute the directions of the tracts to the fined the scalar colors of each tract
    start_time = time.time()
    tract_dir = [compute_direction(trk_n) for trk_n in coord_arr]
    duration = time.time() - start_time
    print("Compute directions: {:.2f} ms".format(1e3*duration))

    # Convert each tracts array to a vtk tube colored by the direction
    start_time = time.time()
    tubes_list = [compute_tubes(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(coord_arr, tract_dir)]
    duration = time.time() - start_time
    print("Create tubes: {:.2f} ms".format(1e3*duration))

    # Convert all the tubes to a single actor
    start_time = time.time()
    actor = visualize_tracts(tubes_list)
    duration = time.time() - start_time
    print("Create actor: {:.2f} ms".format(1e3*duration))

    # Rendering
    start_time = time.time()
    renderer.AddActor(actor)
    ren_win.Render()
    duration = time.time() - start_time
    print("Render: {:.2f} ms".format(1e3*duration))

    # Initialize program
    interactor.Initialize()
    interactor.Start()

    # End program
    ren_win.Finalize()
    interactor.TerminateApp()

# Results for 200 tracts in my computer:
# Convert to array: 3.98 ms
# Compute directions: 5.95 ms
# Create tubes: 93.78 ms
# Create actor: 1.00 ms
# Render: 757.96 ms
