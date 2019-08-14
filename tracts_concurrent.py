import time
import os
import Trekker
import vtk
import numpy as np
import concurrent.futures
from itertools import repeat


def cpu_bound(number):
    return sum(i * i for i in range(number))


# def find_sums(tracker_init, seed):
#     with mp.Pool(5) as pool:
#         # pool.map(cpu_bound, numbers)
#         pool.starmap(visualizeTracks, zip(tracker_init, seed))


# This function converts a single track to a vtkActor
def visualizeTracks(tracker, seed):
    # Input the seed to the tracker object
    tracker.set_seeds(seed)

    # Run the tracker
    # This step will create N tracks if seed is a 3xN matrix
    tractogram = tracker.run()

    # Convert the first track to a vtkActor, i.e., tractogram[0] is the track
    # computed for the first seed
    # return trk2vtkActor(tractogram[0])

    # renderer.AddActor(trkActor)

    # convert trk to vtkPolyData
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

    return trkActor


def main():
    # Initialize a Trekker tracker objects by providing the input FOD image
    # This will just read the image, put in memory
    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    FOD_path = b"sub-P0_dwi_FOD.nii"
    # FOD_path = b"test_fod.nii"
    full_path = os.path.join(data_dir, FOD_path)
    tracker = Trekker.tracker(full_path)

    output1 = list()
    output2 = list()

    start = time.time()
    seed = [np.array([[-8.49, -8.39, 2.5]])] * 2
    tracker_list = [tracker for _ in range(2)]

    # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for out1, out2 in executor.map(visualizeTracks, zip(tracker_list, seed)):
            # put results into correct output list
            output1.append(out1)
            output2.append(out2)

    finish = time.time()
    # these kinds of format strings are only available on Python 3.6:
    # time to upgrade!
    # print(f'original inputs: {repr(output1)}')
    # print(f'total time to execute {sum(output2)} = sum({repr(output2)})')
    # print(f'time saved by parallelizing: {sum(output2) - (finish - start)}')
    # print(f'returned in order given: {repr(output3)}')
    print(output1, output2)


if __name__ == "__main__":
    main()

    # # Create a rendering window, renderer and interactor
    # renderer = vtk.vtkRenderer()
    # renderWindow = vtk.vtkRenderWindow()
    # renderWindow.AddRenderer(renderer)
    # renderWindow.SetSize(640, 480)
    # interactor = vtk.vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(renderWindow)
    #
    # start_time = time.time()
    #
    # duration = time.time() - start_time
    # print(f"Tract computing duration {duration} seconds")
    #
    # start_time = time.time()
    # renderWindow.Render()
    # duration = time.time() - start_time
    # print(f"Render duration {duration} seconds")
    #
    # # Initialize program
    # interactor.Initialize()
    # interactor.Start()
    #
    # # End program
    # renderWindow.Finalize()
    # interactor.TerminateApp()
    #
    # # numbers = [5_000_000 + x for x in range(20)]
    # # find_sums(numbers)