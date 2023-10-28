import concurrent.futures
import queue
import threading
import time
import os
import numpy as np
import vtk
import Trekker


def generate_coord(queue, event):
    # n_tracts, tracker, seed = inp
    print("Start generate_coord\n")
    """Pretend we're getting a number from the network."""
    # print(f"This is event {event.is_set()}")
    while not event.is_set() or not queue.full():
        time.sleep(0.5)
        print("Enter generate_coord\n")
        seed = np.array([[-8.49, -8.39, 2.5]])
        queue.put(seed)
        visualize_coord(seed, "generate_coord")


def visualize_coord(coord, msg):
    # n_tracts, tracker, seed = inp
    # n_tracts, tracker = inp

    print(f"Visualization sent by: {msg} and coord: {coord}\n")


def compute_tracts(inp, queue_coord, queue_tract, event):
    # n_tracts, tracker, seed = inp
    n_tracts, tracker = inp
    print("Start compute_tracts\n")
    """Pretend we're getting a number from the network."""
    # While the event is not set or the queue is not empty
    # or not queue_tract.full()
    while not event.is_set() or not queue_coord.empty() or not queue_tract.full():
    # while not event.is_set():
    #     time.sleep(0.5)
        print("Enter compute_tracts\n")
        trk_list = []
        # seed = queue_coord.get()
        seed = np.array([[-8.49, -8.39, 2.5]])
        tracker.set_seeds(np.repeat(seed, n_tracts, axis=0))
        if tracker.run():
            trk_list.extend(tracker.run())
        queue_tract.put(trk_list)


def split_simple(inp, queue, event):
    """Pretend we're saving a number in the database."""
    print("Start split_simple\n")
    while not event.is_set() or not queue.empty():
    # while not event.is_set():
        # time.sleep(0.5)
        print("Enter split_simple\n")
        # trk_list = queue.get_nowait()
        trk_list = queue.get()
        # print(f"This is the list {trk_list}")
        start_time = time.time()
        trk_arr = [np.asarray(trk_n).T if trk_n else None for trk_n in trk_list]
        duration = time.time() - start_time
        print(f"Tract run trk_arr duration {duration} seconds")

        start_time = time.time()
        trk_dir = [compute_direction(trk_n) for trk_n in trk_arr]
        duration = time.time() - start_time
        print(f"Tract run trk_dir duration {duration} seconds")

        start_time = time.time()
        out_list = [to_vtk(trk_arr_n, trk_dir_n) for trk_arr_n, trk_dir_n in zip(trk_arr, trk_dir)]
        duration = time.time() - start_time
        print(f"Tract run to_vtk duration {duration} seconds")

        # visualize_tracts(out_list, inp, "split_simple")


def compute_direction(trk_n):
    # trk_d = np.diff(trk_n, axis=0, append=2*trk_n[np.newaxis, -1, :])
    trk_d = np.diff(trk_n, axis=0, append=trk_n[np.newaxis, -2, :])
    trk_d[-1, :] *= -1
    # check that linalg norm makes second norm
    # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    direction = 255 * np.absolute((trk_d / np.linalg.norm(trk_d, axis=1)[:, None]))
    return direction.astype(int)


def to_vtk(trk, direc):
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
        colors.InsertNextTuple(direc[j, :])
        k += 1

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


def visualize_tracts(out_list, inp, msg):
    renderer, renderWindow = inp
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

    # render_actors(actor, renderer, interactor)
    renderer.AddActor(actor)
    renderWindow.Render()

    print(f"Visualization sent by: {msg}")

    # # Initialize program
    # interactor.Initialize()
    # interactor.Start()
    #
    # # End program
    # renderWindow.Finalize()
    # interactor.TerminateApp()


if __name__ == "__main__":

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
    tracker.set_numberOfThreads(8)
    n_tracts = 5
    # seed = np.array([[-8.49, -8.39, 2.5]])
    # seed_coord = np.array([[-8.49, -8.39, 2.5]])
    # tracker.set_seeds(np.repeat(seed_coord, 4, axis=0))
    duration = time.time() - start_time
    print(f"Initialize Trekker duration {duration} seconds")

    # Create a rendering window, renderer and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(640, 480)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    pipe_coord = queue.LifoQueue(maxsize=1)
    pipe_tract = queue.LifoQueue(maxsize=1)
    event = threading.Event()
    # inp_trk = (n_tracts, tracker, seed)
    inp_trk = (n_tracts, tracker)
    inp_ren = renderer, renderWindow
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        print("About start generate_coord\n")
        executor.submit(generate_coord, pipe_coord, event)
        print("About start compute_tract\n")
        executor.submit(compute_tracts, inp_trk, pipe_coord, pipe_tract, event)
        print("About start split_simple\n")
        executor.submit(split_simple, inp_ren, pipe_tract, event)
        time.sleep(10)
        # print("setting event")
        event.set()

    # start_time = time.time()
    # actor = visualize_tracts(tube_list)
    # duration = time.time() - start_time
    # print(f"Visualize duration {duration} seconds")

