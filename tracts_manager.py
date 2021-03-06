#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import Trekker
import vtk
import numpy as np
import multiprocessing as mp


def visualizeTracks(tracker, seed):

    tracker.set_seeds(seed)

    tractogram = tracker.run()

    trk = np.transpose(np.asarray(tractogram[0]))
    numberOfPoints = trk.shape[0]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    colors = vtk.vtkFloatArray()
    colors.SetNumberOfComponents(4)
    colors.SetName("tangents")

    k = 0
    lines.InsertNextCell(numberOfPoints)
    for j in range(numberOfPoints):
        points.InsertNextPoint(trk[j, :])
        lines.InsertCellPoint(k)
        k = k + 1

        if j < (numberOfPoints - 1):
            direction = trk[j + 1, :] - trk[j, :]
            direction = direction / np.linalg.norm(direction)
            colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))
        else:
            colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))

    trkData = vtk.vtkPolyData()
    trkData.SetPoints(points)
    trkData.SetLines(lines)
    trkData.GetPointData().SetScalars(colors)

    # make it a tube
    trkTube = vtk.vtkTubeFilter()
    trkTube.SetRadius(0.1)
    trkTube.SetNumberOfSides(4)
    trkTube.SetInputData(trkData)
    trkTube.Update()

    # mapper
    trkMapper = vtk.vtkPolyDataMapper()
    trkMapper.SetInputData(trkTube.GetOutput())

    # actor
    trkActor = vtk.vtkActor()
    trkActor.SetMapper(trkMapper)

    trk_act.append(trkActor)

    # return trkActor


def main():
    manager = mp.Manager()
    global tracker
    global trk_act

    # Initialize a Trekker tracker objects by providing the input FOD image
    # This will just read the image, put in memory
    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    FOD_path = b"sub-P0_dwi_FOD.nii"
    # FOD_path = b"test_fod.nii"
    full_path = os.path.join(data_dir, FOD_path)
    tracker = manager.Trekker.tracker(full_path)
    trk_act = manager.list()

    # Create a rendering window, renderer and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    # Show tracks
    # for i in range(5):
    seed = [np.array([[-8.49, -8.39, 2.5]])]*5
    # tracker_list = [tracker for _ in range(5)]

    start_time = time.time()

    pool = mp.Pool(5)
    # pool.map(cpu_bound, numbers)
    pool.map(visualizeTracks, seed)
    pool.close()
    pool.join()

    duration = time.time() - start_time
    print(f"Tract computing duration {duration} seconds")

    start_time = time.time()
    renderWindow.Render()
    duration = time.time() - start_time
    print(f"Render duration {duration} seconds")

    # Initialize program
    interactor.Initialize()
    interactor.Start()

    # End program
    renderWindow.Finalize()
    interactor.TerminateApp()


if __name__ == "__main__":
    main()
