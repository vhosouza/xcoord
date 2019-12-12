import time

import Trekker
import numpy as np
import os
import vtk

import pickle
import threading
import math


"""
Thread to update the coordinates with the fiducial points
co-registration method while the Navigation Button is pressed.
Sleep function in run method is used to avoid blocking GUI and
for better real-time navigation
"""


class ComputeTracts(threading.Thread):
    """
    Thread to update the coordinates with the fiducial points
    co-registration method while the Navigation Button is pressed.
    Sleep function in run method is used to avoid blocking GUI and
    for better real-time navigation
    """

    def __init__(self, tracker, position, n_tracts):
        threading.Thread.__init__(self)
        # trekker variables
        self.tracker = tracker
        self.position = position
        self.n_tracts = n_tracts
        # threading variable
        self._pause_ = False
        # self.mutex = threading.Lock()
        # self.start()

    def stop(self):
        # self.mutex.release()
        self._pause_ = True

    def run(self):
        if self._pause_:
            return
        else:
            # self.mutex.acquire()
            try:
                seed = self.position

                chunck_size = 10
                nchuncks = math.floor(self.n_tracts / chunck_size)
                # print("The chunck_size: ", chunck_size)
                # print("The nchuncks: ", nchuncks)

                root = vtk.vtkMultiBlockDataSet()
                # n = 1
                n_tracts = 0
                # while n <= nchuncks:
                for n in range(nchuncks):
                    # Compute the tracts
                    trk_list = []
                    # for _ in range(chunck_size):
                    self.tracker.set_seeds(np.repeat(seed, chunck_size, axis=0))
                    if self.tracker.run():
                        trk_list.extend(self.tracker.run())

                    # Transform tracts to array
                    trk_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in trk_list]

                    # Compute the directions
                    trk_dir = [simple_direction(trk_n) for trk_n in trk_arr]

                    # Compute the vtk tubes
                    out_list = [compute_tubes_vtk(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(trk_arr, trk_dir)]
                    # Compute the actor
                    root = tracts_root(out_list, root, n_tracts)
                    n_tracts += len(out_list)

                    # wx.CallAfter(Publisher.sendMessage, 'Update tracts', flag=True, root=root, affine_vtk=self.affine_vtk)
            finally:
                self.mutex.release()

                # time.sleep(0.05)
                # n += 1


def simple_direction(trk_n):
    # trk_d = np.diff(trk_n, axis=0, append=2*trk_n[np.newaxis, -1, :])
    trk_d = np.diff(trk_n, axis=0, append=trk_n[np.newaxis, -2, :])
    trk_d[-1, :] *= -1
    # check that linalg norm makes second norm
    # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    direction = 255 * np.absolute((trk_d / np.linalg.norm(trk_d, axis=1)[:, None]))
    return direction.astype(int)


def compute_tubes(trk_list):
    # start_time = time.time()

    trk = np.transpose(np.asarray(trk_list))
    numb_points = trk.shape[0]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # colors = vtk.vtkFloatArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    # colors.SetName("tangents")

    lines.InsertNextCell(numb_points)
    k = 0
    direc_out = []
    for j in range(numb_points):
        points.InsertNextPoint(trk[j, :])
        lines.InsertCellPoint(k)
        k = k + 1

        if j < (numb_points - 1):
            direction = trk[j + 1, :] - trk[j, :]
            direction = direction / np.linalg.norm(direction)
            direction_rescale = [int(255*abs(s)) for s in direction]
            # colors.InsertNextTuple(np.abs([direc[0], direc[1], direc[2], 1]))
            colors.InsertNextTuple(direction_rescale)
        else:
            # colors.InsertNextTuple(np.abs([direc[0], direc[1], direc[2], 1]))
            colors.InsertNextTuple(direction_rescale)
        direc_out.append(direction_rescale)

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

    # duration = time.time() - start_time
    # print(f"Tube computing duration {duration} seconds")

    return trkTube, direc_out


def compute_tubes_vtk(trk, direc):
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
        k += 1

        # if j < (numb_points - 1):
        colors.InsertNextTuple(direc[j, :])
        # else:
        #     colors.InsertNextTuple(direc[j, :])

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

    return trkTube


def single_process(trk_list):
    out_list = []
    for n, trk_n in enumerate(trk_list):
        tube, direct = compute_tubes(trk_n)
        out_list.append(tube)
        # out_list.append(compute_tubes(trk_n))
    return out_list


def list_loop(trk_list):
    out_list = [compute_tubes(trk_n) if trk_n else None for trk_n in trk_list]
    return out_list


def split_simple(trk_list):
    start_time = time.time()
    trk_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in trk_list]
    duration = time.time() - start_time
    print(f"Tract run trk_arr duration {duration} seconds")

    start_time = time.time()
    trk_dir = [simple_direction(trk_n) for trk_n in trk_arr]
    duration = time.time() - start_time
    print(f"Tract run trk_dir duration {duration} seconds")

    start_time = time.time()
    out_list = [to_vtk(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(trk_arr, trk_dir)]
    duration = time.time() - start_time
    print(f"Tract run to_vtk duration {duration} seconds")

    return out_list


def compute_tracts(n_tracts, tracker, seed):
    # trk_list = [None] * n_tracts
    trk_list = []
    # for n in range(n_tracts):
        # tracker.set_seeds(seed)
    tracker.set_seeds(np.repeat(seed, n_tracts, axis=0))
        # trk_run = tracker.run()
        # trk_list[n] = tracker.run()
    if tracker.run():
        # trk_list.append(tracker.run()[0])
        trk_list.extend(tracker.run())
    return trk_list


def tracts_root(out_list):
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


if __name__ == "__main__":
    save_id = True

    if save_id:
        start_time = time.time()
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
        tracker.set_numberOfThreads(10)
        n_tracts = 100
        seed = np.array([[-8.49, -8.39, 2.5]])
        # seed_coord = np.array([[-8.49, -8.39, 2.5]])
        # tracker.set_seeds(np.repeat(seed_coord, 4, axis=0))
        duration = time.time() - start_time
        print(f"Initialize Trekker duration {duration} seconds")

        start_time = time.time()
        trk_list = compute_tracts(n_tracts, tracker, seed)
        duration = time.time() - start_time
        print(f"Tract run duration {duration} seconds")

        with open('track_list_parallel.trk', 'wb') as fp:
            pickle.dump(trk_list, fp)

    else:
        with open('track_list_parallel.trk', 'rb') as fp:
            trk_list = pickle.load(fp)

    seed = np.array([[-8.49, -8.39, 2.5]])
    print("Seed: ", seed)

    # Create a rendering window, renderer and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    start_time = time.time()
    final_list = single_process(trk_list)
    duration = time.time() - start_time
    print(f"Tract computing singleprocess duration {duration} seconds")

    start_time = time.time()
    final_list = list_loop(trk_list)
    duration = time.time() - start_time
    print(f"Tract computing listloop duration {duration} seconds")

    start_time = time.time()
    final_list_2 = split_simple(trk_list)
    duration = time.time() - start_time
    print(f"Tract computing split_simple duration {duration} seconds")

    # start_time = time.time()
    # final_list_3 = threading_process(trk_list)
    # duration = time.time() - start_time
    # print(f"Tract computing threading_process duration {duration} seconds")

    # Extremely slow (82 seconds for 200 tracts)
    # start_time = time.time()
    # final_list_4 = multi_process_2(trk_list)
    # duration = time.time() - start_time
    # print(f"Tract computing multiprocess_2 duration {duration} seconds")

    # start_time = time.time()
    # final_list_m = multi_process(trk_list)
    # duration = time.time() - start_time
    # print(f"Tract computing multiprocess duration {duration} seconds")

    # start_time = time.time()
    # final_list_n = compute_direction_numba(trk_list)
    # duration = time.time() - start_time
    # print(f"Tract computing numba duration {duration} seconds")

    start_time = time.time()
    actor = tracts_root(final_list_2)
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

#Result for 200 tracts:
# There is a 5x improvement in speed in using split_simple
# Initialize Trekker duration 2.336592435836792 seconds
# Tract run duration 66.09394478797913 seconds
# Tract computing singleprocess duration 0.5225706100463867 seconds
# Tract computing listloop duration 0.5585072040557861 seconds
# Tract run trk_arr duration 0.003989219665527344 seconds
# Tract run trk_dir duration 0.005984306335449219 seconds
# Tract run to_vtk duration 0.09275221824645996 seconds
# Tract computing split_simple duration 0.10272574424743652 seconds
# Tract computing threading_process duration 0.14174294471740723 seconds
# Visualize duration 0.0009970664978027344 seconds
# Render duration 0.7962656021118164 seconds