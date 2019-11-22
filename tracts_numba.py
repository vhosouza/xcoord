#import threading

import multiprocessing as mp

import time

import Trekker
import numpy as np
import os
import vtk

"""
Thread to update the coordinates with the fiducial points
co-registration method while the Navigation Button is pressed.
Sleep function in run method is used to avoid blocking GUI and
for better real-time navigation
"""


def visualize_tracts(out_list):
    # create tracts only when at least one was computed
    if not out_list.count(None) == len(out_list):
        root = vtk.vtkMultiBlockDataSet()

        for n, tube in enumerate(out_list):
            if tube:
                root.SetBlock(n, tube.GetOutput())

        # https://lorensen.github.io/VTKExamples/site/Python/CompositeData/CompositePolyDataMapper/
        mapper = vtk.vtkCompositePolyDataMapper2()
        mapper.SetInputDataObject(root)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

    return actor
        #actor.SetUserMatrix(self.affine_vtk)
        # duration = time.time() - start_time
        # print(f"Tract computing duration {duration} seconds")


def multiprocess_test(n_tracts, tracker, seed):
    data = [tracker]*n_tracts
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(compute_tracts, args=(n_tracts, row, seed)) for row in data]

    # Step 3: Don't forget to close
    pool.close()
    return results


def trk2vtkActor(tracker, seed):
    start_time = time.time()
    tracker.set_seeds(seed)
    # convert trk to vtkPolyData
    trk_run = tracker.run()
    duration = time.time() - start_time
    print(f"Tract run duration {duration} seconds")

    start_time = time.time()
    if trk_run:
        trk = np.transpose(np.asarray(trk_run[0]))
        numb_points = trk.shape[0]

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        # colors = vtk.vtkFloatArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        # colors.SetName("tangents")

        k = 0
        lines.InsertNextCell(numb_points)
        for j in range(numb_points):
            points.InsertNextPoint(trk[j, :])
            lines.InsertCellPoint(k)
            k = k + 1

            if j < (numb_points - 1):
                direction = trk[j + 1, :] - trk[j, :]
                direction = direction / np.linalg.norm(direction)
                direc = [int(255*abs(s)) for s in direction]
                # colors.InsertNextTuple(np.abs([direc[0], direc[1], direc[2], 1]))
                colors.InsertNextTuple(direc)
            else:
                # colors.InsertNextTuple(np.abs([direc[0], direc[1], direc[2], 1]))
                colors.InsertNextTuple(direc)

        trkData = vtk.vtkPolyData()
        trkData.SetPoints(points)
        trkData.SetLines(lines)
        trkData.GetPointData().SetScalars(colors)

        # make it a tube
        trkTube = vtk.vtkTubeFilter()
        trkTube.SetRadius(0.5)
        trkTube.SetNumberOfSides(4)
        trkTube.SetInputData(trkData)
        trkTube.Update()

        duration = time.time() - start_time
        print(f"Tract VTK duration {duration} seconds")

    else:
        trkTube = None

    return trkTube


def compute_tracts(n_tracts, tracker, seed):

    out_list = [None] * n_tracts

    for n in range(n_tracts):
        # print("out_list: ", out_list)
        out_list[n] = trk2vtkActor(tracker, seed)
        # print("out_list {} and seed {}".format(out_list, seed))

    return out_list


if __name__ == "__main__":
    # Initialize a Trekker tracker objects by providing the input FOD image
    # This will just read the image, put in memory
    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    FOD_path = b"sub-P0_dwi_FOD.nii"
    # FOD_path = b"test_fod.nii"
    full_path = os.path.join(data_dir, FOD_path)
    tracker = Trekker.tracker(full_path)
    tracker.set_seed_maxTrials(1)
    tracker.set_stepSize(0.1)
    tracker.set_minFODamp(0.04)
    tracker.set_probeQuality(3)
    n_tracts = 10

    seed = np.array([[-8.49, -8.39, 2.5]])

    # Create a rendering window, renderer and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    start_time = time.time()
    out_list = compute_tracts(n_tracts, tracker, seed)
    duration = time.time() - start_time
    print(f"Tract computing singleprocess duration {duration} seconds")

    start_time = time.time()
    out_list = multiprocess_test(n_tracts, tracker, seed)
    duration = time.time() - start_time
    print(f"Tract computing multiprocess duration {duration} seconds")

    start_time = time.time()
    actor = visualize_tracts(out_list)
    duration = time.time() - start_time
    print(f"Visualize duration {duration} seconds")

    # render_actors(actor, renderer, interactor)
    start_time = time.time()
    renderer.AddActor(actor)
    renderWindow.Render()
    duration = time.time() - start_time
    print(f"Render duration {duration} seconds")

    # Initialize program
    interactor.Initialize()
    interactor.Start()

    # End program
    renderWindow.Finalize()
    interactor.TerminateApp()
