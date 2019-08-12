#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np
import transformations as tf
import Trekker
import vtk


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True

    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'wm_orig_smooth_world.stl'
    brain_path = os.path.join(data_dir, stl_path)

    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'wm.stl'
    brain_inv_path = os.path.join(data_dir, stl_path)

    nii_path = b'sub-P0_dwi_FOD.nii'
    trk_path = os.path.join(data_dir, nii_path)

    nii_path = b'sub-P0_T1w_biascorrected.nii'
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

    tracker = Trekker.tracker(trk_path)

    repos = [0., 0., 0., 0., 0., 0.]
    brain_actor = load_stl(brain_inv_path, ren, opacity=.1, colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.identity(4))
    bds = brain_actor.GetBounds()
    print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    repos = [0., 0., 0., 0., 0., 0.]
    brain_actor_mri = load_stl(brain_path, ren, opacity=.1, colour=[0.0, 1.0, 0.0], replace=repos, user_matrix=np.linalg.inv(affine))
    bds = brain_actor_mri.GetBounds()
    print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

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
    for i in range(5):
        seed = np.array([[-8.49, -8.39, 2.5]])
        visualizeTracks(ren, ren_win, tracker, seed, replace=repos_trk, user_matrix=np.linalg.inv(affine))

    # Assign actor to the renderer
    ren.AddActor(brain_actor)
    ren.AddActor(brain_inv_actor)
    ren.AddActor(brain_actor_mri)

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


def visualizeTracks(renderer, renderWindow, tracker, seed, replace, user_matrix):
    # Input the seed to the tracker object
    tracker.set_seeds(seed)

    # Run the tracker
    # This step will create N tracks if seed is a 3xN matrix
    tractogram = tracker.run()

    # Convert the first track to a vtkActor, i.e., tractogram[0] is the track
    # computed for the first seed
    trkActor = trk2vtkActor(tractogram[0], replace)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    trkActor.SetUserMatrix(matrix_vtk)

    renderer.AddActor(trkActor)
    renderWindow.Render()

    return


# This function converts a single track to a vtkActor
def trk2vtkActor(trk, replace):
    # convert trk to vtkPolyData
    trk = np.transpose(np.asarray(trk))
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
        transform_filt.SetInputConnection(trkTube.GetOutputPort())
        transform_filt.Update()

    # mapper
    trkMapper = vtk.vtkPolyDataMapper()
    trkMapper.SetInputData(transform_filt.GetOutput())

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
