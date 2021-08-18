#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np
import transformations as tf
import Trekker
import vtk
import time
import psutil
import dti_funcs as dti


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True
    n_tracts = 240
    n_threads = 2*psutil.cpu_count()

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\baran\pilot_20200131'
    data_dir = data_dir.encode('utf-8')
    # FOD_path = 'Baran_FOD.nii'
    # trk_path = os.path.join(data_dir, FOD_path)

    # data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'wm_orig_smooth_world.stl'
    brain_path = os.path.join(data_dir, stl_path)

    # data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'gm.stl'
    brain_inv_path = os.path.join(data_dir, stl_path)

    nii_path = b'Baran_FOD.nii'
    trk_path = os.path.join(data_dir, nii_path)

    nii_path = b'Baran_T1_inFODspace.nii'
    img_path = os.path.join(data_dir, nii_path)

    imagedata = nb.squeeze_image(nb.load(img_path.decode('utf-8')))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    # print(imagedata.header)

    print("pix_dim: {}, img_shape: {}".format(pix_dim, img_shape))

    if AFFINE_IMG:
        affine = imagedata.affine
        if NO_SCALE:
            scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
            affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)
    else:
        affine = np.identity(4)

    print("affine: {0}\n".format(affine))

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    start_time = time.time()
    tracker = Trekker.initialize(trk_path)
    tracker.seed_maxTrials(1)
    tracker.minFODamp(0.1)
    tracker.writeInterval(50)
    tracker.maxLength(200)
    tracker.minLength(20)
    tracker.maxSamplingPerStep(100)
    tracker.numberOfThreads(n_threads)
    duration = time.time() - start_time
    print("Initialize Trekker: {:.2f} ms".format(1e3*duration))

    repos = [0., 0., 0., 0., 0., 0.]
    brain_actor = load_stl(brain_inv_path, ren, opacity=.1, colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.identity(4))
    bds = brain_actor.GetBounds()
    print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    # repos = [0., 0., 0., 0., 0., 0.]
    # brain_actor_mri = load_stl(brain_path, ren, opacity=.1, colour=[0.0, 1.0, 0.0], replace=repos, user_matrix=np.linalg.inv(affine))
    # bds = brain_actor_mri.GetBounds()
    # print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    repos = [0., 256., 0., 0., 0., 0.]
    # brain_inv_actor = load_stl(brain_inv_path, ren, colour="SkinColor", opacity=0.5, replace=repos, user_matrix=np.linalg.inv(affine))
    brain_inv_actor = load_stl(brain_inv_path, ren, colour="SkinColor", opacity=.1, replace=repos)

    # Add axes to scene origin
    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Show tracks
    repos_trk = [0., -256., 0., 0., 0., 0.]

    matrix_vtk = vtk.vtkMatrix4x4()

    trans = np.identity(4)
    trans[1, -1] = repos_trk[1]
    final_matrix = np.linalg.inv(affine) @ trans

    print("final_matrix: {}".format(final_matrix))

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, final_matrix[row, col])

    root = vtk.vtkMultiBlockDataSet()
    # for i in range(10):
        # seed = np.array([[-8.49, -8.39, 2.5]])
    seed = np.array([[27.53, -77.37, 46.42]])

    tracts_actor = dti.single_block(tracker, seed, n_tracts, root, matrix_vtk)

    # out_list = []
    count_tracts = 0
    start_time_all = time.time()

    for n in range(round(n_tracts/n_threads)):
        branch = dti.multi_block(tracker, seed, n_threads)
        count_tracts += branch.GetNumberOfBlocks()

        # start_time = time.time()
        # root = dti.tracts_root(out_list, root, n)
        root.SetBlock(n, branch)
        # duration = time.time() - start_time
        # print("Compute root {}: {:.2f} ms".format(n, 1e3*duration))

    duration = time.time() - start_time_all
    print("Compute multi {}: {:.2f} ms".format(n, 1e3*duration))
    print("Number computed tracts {}".format(count_tracts))
    print("Number computed branches {}".format(root.GetNumberOfBlocks()))

    start_time = time.time()
    tracts_actor = dti.compute_actor(root, matrix_vtk)
    duration = time.time() - start_time
    print("Compute actor: {:.2f} ms".format(1e3*duration))

    # Assign actor to the renderer
    ren.AddActor(brain_actor)
    ren.AddActor(brain_inv_actor)

    start_time = time.time()
    ren.AddActor(tracts_actor)
    duration = time.time() - start_time
    print("Add actor: {:.2f} ms".format(1e3*duration))
    # ren.AddActor(brain_actor_mri)

    # Enable user interface interactor
    iren.Initialize()
    ren_win.Render()
    iren.Start()


def load_stl(stl_path, ren, opacity=1., visibility=1, position=False, colour=False, replace=False, user_matrix=np.identity(4)):
    vtk_colors = vtk.vtkNamedColors()
    vtk_colors.SetColor("SkinColor", [233, 200, 188, 255])
    vtk_colors.SetColor("BkgColor", [51, 77, 102, 255])

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    poly_normals = vtk.vtkPolyDataNormals()
    poly_normals.SetInputData(reader.GetOutput())
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.SplittingOff()
    poly_normals.Update()

    if replace:
        transx, transy, transz, rotx, roty, rotz = replace
        # create a transform that rotates the stl source
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateX(rotx)
        transform.RotateY(roty)
        transform.RotateZ(rotz)
        transform.Translate(transx, transy, transz)

        transform_filt = vtk.vtkTransformPolyDataFilter()
        transform_filt.SetTransform(transform)
        transform_filt.SetInputConnection(poly_normals.GetOutputPort())
        transform_filt.Update()

    mapper = vtk.vtkPolyDataMapper()

    if vtk.VTK_MAJOR_VERSION <= 5:
        if replace:
            mapper.SetInput(transform_filt.GetOutput())
        else:
            mapper.SetInput(poly_normals.GetOutput())
    else:
        if replace:
            mapper.SetInputConnection(transform_filt.GetOutputPort())
        else:
            mapper.SetInputConnection(poly_normals.GetOutputPort())

    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.SetVisibility(visibility)
    actor.GetProperty().SetBackfaceCulling(1)

    if colour:
        if type(colour) is str:
            actor.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d("SkinColor"))
            actor.GetProperty().SetSpecular(.3)
            actor.GetProperty().SetSpecularPower(20)

        else:
            actor.GetProperty().SetColor(colour)

    if position:
        actor.SetPosition(position)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    actor.SetUserMatrix(matrix_vtk)

    # Assign actor to the renderer
    ren.AddActor(actor)

    return actor


def visualizeTracks(renderer, renderWindow, tracker, seed, user_matrix):
    # Input the seed to the tracker object
    tracker.seed_coordinates(np.repeat(seed, 200, axis=0))

    # Run the tracker
    # This step will create N tracks if seed is a 3xN matrix
    tractogram = tracker.run()

    # Convert the first track to a vtkActor, i.e., tractogram[0] is the track
    # computed for the first seed
    trkActor = trk2vtkActor(tractogram[0])

    trkActor.SetUserMatrix(user_matrix)

    renderer.AddActor(trkActor)
    renderWindow.Render()

    return


# This function converts a single track to a vtkActor
def trk2vtkActor(trk):
    # convert trk to vtkPolyData
    trk = np.transpose(np.asarray(trk))
    numberOfPoints = trk.shape[0]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    # colors = vtk.vtkFloatArray()
    # colors.SetNumberOfComponents(4)
    # colors.SetName("tangents")

    k = 0
    lines.InsertNextCell(numberOfPoints)
    for j in range(numberOfPoints):
        points.InsertNextPoint(trk[j, :])
        lines.InsertCellPoint(k)
        k = k + 1

        if j < (numberOfPoints - 1):
            direction = trk[j + 1, :] - trk[j, :]
            direction = direction / np.linalg.norm(direction)
            direc = [int(255 * abs(s)) for s in direction]
            colors.InsertNextTuple(direc)
            # colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))
        else:
            colors.InsertNextTuple(direc)
            # colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))

    trkData = vtk.vtkPolyData()
    trkData.SetPoints(points)
    trkData.SetLines(lines)
    trkData.GetPointData().SetScalars(colors)

    # make it a tube
    trkTube = vtk.vtkTubeFilter()
    trkTube.SetRadius(0.3)
    trkTube.SetNumberOfSides(4)
    trkTube.SetInputData(trkData)
    trkTube.Update()

    # if replace:
    #     transx, transy, transz, rotx, roty, rotz = replace
    #     # create a transform that rotates the stl source
    #     transform = vtk.vtkTransform()
    #     transform.PostMultiply()
    #     transform.RotateX(rotx)
    #     transform.RotateY(roty)
    #     transform.RotateZ(rotz)
    #     transform.Translate(transx, transy, transz)
    #
    #     transform_filt = vtk.vtkTransformPolyDataFilter()
    #     transform_filt.SetTransform(transform)
    #     transform_filt.SetInputConnection(trkTube.GetOutputPort())
    #     transform_filt.Update()

    # mapper
    trkMapper = vtk.vtkPolyDataMapper()
    trkMapper.SetInputData(trkTube.GetOutput())

    # actor
    trkActor = vtk.vtkActor()
    trkActor.SetMapper(trkMapper)

    return trkActor


def add_line(renderer, p1, p2, color=[0.0, 0.0, 1.0]):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)

    renderer.AddActor(actor)


if __name__ == '__main__':
    main()

# 240 tracts in a single block/run
# Seed coordinates: 0.00 ms
# Run Trekker: 1154.00 ms
# Tracts to array: 2.00 ms
# Tracts directions: 12.00 ms
# Compute tubes: 59.00 ms
# Compute root: 1.00 ms
# Tracts computation: 80.00 ms
# Compute actor: 0.00 ms
# Add actor: 0.00 ms
