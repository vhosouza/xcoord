#!/usr/bin/env python
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

import nibabel as nb
from nibabel.affines import apply_affine
import numpy as np
import vtk
import os

import load_coords as lc
import nexstim2mri as n2m


def main():

    SHOW_AXES = True
    SHOW_COIL = True
    SHOW_SKIN = False
    SHOW_BRAIN = True

    reorder = [0, 2, 1]
    flipx = [True, False, False]

    data_dir = os.environ['OneDriveConsumer'] + '\\data\\nexstim_coord\\'
    filename_coord = data_dir + 's01_eximia_coords.txt'
    img_path = data_dir + 'ppM1_S1.nii'
    coil_path = data_dir + 'magstim_fig8_coil.stl'

    coords = lc.load_nexstim(filename_coord)
    # red, gree, blue, maroon (dark red), olive (shitty green), teal (petrol blue), yellow, orange
    col = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.5, .0, 0.], [.5, .5, 0.], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

    imagedata = nb.squeeze_image(nb.load(img_path))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    # hdr = imagedata.header
    img_shape = imagedata.header.get_data_shape()
    affine = imagedata.affine

    # print(data_dir)
    # [print(s) for s in coords]

    # brain_file = data_dir + "s01_skin_inv.stl"
    brain_file = data_dir + "s01_gm_nc.stl"
    skin_file = data_dir + "s01_skin_nc.stl"

    brain = vtk.vtkSTLReader()
    brain.SetFileName(brain_file)

    skin = vtk.vtkSTLReader()
    skin.SetFileName(skin_file)

    # brain.Update()
    # polydata = brain.GetOutput()
    #
    # print('NUMBER OF POINTS: ', polydata.GetNumberOfPoints())

    brain_mapper = vtk.vtkPolyDataMapper()
    skin_mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        brain_mapper.SetInput(brain.GetOutput())
        skin_mapper.SetInput(skin.GetOutput())
    else:
        brain_mapper.SetInputConnection(brain.GetOutputPort())
        skin_mapper.SetInputConnection(skin.GetOutputPort())

    brain_actor = vtk.vtkActor()
    brain_actor.SetMapper(brain_mapper)
    brain_actor.GetProperty().SetColor(0., 1., 1.)
    brain_actor.GetProperty().SetOpacity(1.)
    if SHOW_BRAIN:
        brain_actor.SetVisibility(1)
    else:
        brain_actor.SetVisibility(0)

    skin_actor = vtk.vtkActor()
    skin_actor.SetMapper(skin_mapper)
    skin_actor.GetProperty().SetOpacity(.7)
    if SHOW_SKIN:
        skin_actor.SetVisibility(1)
    else:
        skin_actor.SetVisibility(0)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Assign actor to the renderer
    ren.AddActor(brain_actor)
    ren.AddActor(skin_actor)

    # MRI landmarks
    # coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Coil Loc'], ['EF max']]
    # pts_ref = [1, 2, 3, 7, 10]
    # all coords
    # coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Nose/Nasion'], ['Left ear'], ['Right ear'],
    #              ['Coil Loc'], ['EF max']]
    # pts_ref = [1, 2, 3, 5, 4, 6, 7, 10]
    # scalp landmarkds
    coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Coil Loc'], ['EF max']]
    pts_ref = [5, 4, 6, 7, 10]

    for n, pts_id in enumerate(pts_ref):
        coord_aux = n2m.coord_change(coords[pts_id][1:], img_shape, affine, flipx, reorder)
        [coord_mri[n].append(s) for s in coord_aux]

        act = add_marker(coord_aux, col[n])
        ren.AddActor(act)

    print('coords:\n', coords)
    print('coords_mri:\n', coord_mri)

    # coil_loc_mri = n2m.coord_change(img_shape, coords[7][1:])
    # coil_loc = n2m.coord_change(coords[7][1:], img_shape, affine, flipx, reorder)
    coil_loc = coord_mri[-2][1:]

    coil_norm = n2m.coord_change(coords[8][1:], img_shape, affine, flipx, reorder)
    coil_norm = coords[8][1:]

    coil_dir = n2m.coord_change(coords[9][1:], img_shape, affine, flipx, reorder)
    coil_dir = coords[9][1:]


    # act_coil = create_plane(coil_loc, coil_dir, coil_norm)
    # act_coil = create_coil(coil_path, coil_loc, coil_dir, coil_norm)
    # ren.AddActor(act_coil)

    axes = vtk.vtkAxesActor()

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(iren)
    # widget.SetViewport(0.0, 0.0, 0.4, 0.4)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    if SHOW_COIL:
        p1 = coords[7][1:]
        p2 = [x + 50*y for x, y in zip(p1, coil_norm)]
        p2_norm = n2m.coord_change(p2, img_shape, affine, flipx, reorder)
        add_line(ren, coil_loc, p2_norm, color=[.0, 1.0, 0.0])
        p2 = [x + 50*y for x, y in zip(p1, coil_dir)]
        p2_dir = n2m.coord_change(p2, img_shape, affine, flipx, reorder)
        add_line(ren, coil_loc, p2_dir, color=[1.0, .0, .0])
        coil_face = np.cross(coil_dir, coil_norm)
        p2 = [x + 50 * y for x, y in zip(p1, coil_face.tolist())]
        p2_face = n2m.coord_change(p2, img_shape, affine, flipx, reorder)
        add_line(ren, coil_loc, p2_face, color=[.0, .0, 1.0])

    # Enable user interface interactor
    iren.Initialize()
    ren.ResetCamera()
    renWin.Render()
    iren.Start()


def add_marker(coord, color):
    x, y, z = coord

    ball_ref = vtk.vtkSphereSource()
    ball_ref.SetRadius(2)
    ball_ref.SetCenter(x, y, z)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ball_ref.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(color)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    return actor


def create_plane(coil_center, coil_dir, coil_normal):

    coil_plane = vtk.vtkPlaneSource()
    coil_plane.SetOrigin(0, 0, 0)
    coil_plane.SetPoint1(100, 0, 0)
    coil_plane.SetPoint2(0, 100, 0)
    # coil_plane.SetCenter(coil_center)
    # coil_plane.SetNormal(coil_normal)
    coil_plane.SetCenter(0., 0., 0.)
    coil_plane.SetNormal(0., 0., 1.)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(coil_plane.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(0.5, 0., 0.5)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    return actor


def load_stl(stl_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)

    print(stl_path)

    transform = vtk.vtkTransform()
    # transform.RotateZ(90)
    transform.RotateZ(0)

    transform_filt = vtk.vtkTransformPolyDataFilter()
    transform_filt.SetTransform(transform)
    transform_filt.SetInputData(reader.GetOutput())
    transform_filt.Update()

    normals = vtk.vtkPolyDataNormals()
    # normals.SetInputData(transform_filt.GetOutput())
    normals.SetInputData(reader.GetOutput())
    normals.SetFeatureAngle(80)
    normals.AutoOrientNormalsOn()
    normals.Update()

    obj_mapper = vtk.vtkPolyDataMapper()
    obj_mapper.SetInputConnection(reader.GetOutputPort())
    # obj_mapper.SetInputData(normals.GetOutput())
    # obj_mapper.ScalarVisibilityOff()
    # obj_mapper.ImmediateModeRenderingOn()  # improve performance

    stl_actor = vtk.vtkActor()
    stl_actor.SetMapper(obj_mapper)
    # coil_actor.GetProperty().SetOpacity(0.9)
    stl_actor.SetVisibility(1)
    # coil_actor.SetUserMatrix(m_img_vtk)

    return stl_actor


def create_coil(coil_path, coil_center, coil_dir, coil_normal):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(coil_path)

    print(coil_path)

    transform = vtk.vtkTransform()
    # transform.RotateZ(90)
    transform.RotateZ(0)

    transform_filt = vtk.vtkTransformPolyDataFilter()
    transform_filt.SetTransform(transform)
    transform_filt.SetInputData(reader.GetOutput())
    transform_filt.Update()

    normals = vtk.vtkPolyDataNormals()
    # normals.SetInputData(transform_filt.GetOutput())
    normals.SetInputData(reader.GetOutput())
    normals.SetFeatureAngle(80)
    normals.AutoOrientNormalsOn()
    normals.Update()

    obj_mapper = vtk.vtkPolyDataMapper()
    obj_mapper.SetInputConnection(reader.GetOutputPort())
    # obj_mapper.SetInputData(normals.GetOutput())
    # obj_mapper.ScalarVisibilityOff()
    # obj_mapper.ImmediateModeRenderingOn()  # improve performance

    coil_actor = vtk.vtkActor()
    coil_actor.SetMapper(obj_mapper)
    # coil_actor.GetProperty().SetOpacity(0.9)
    coil_actor.SetVisibility(1)
    # coil_actor.SetUserMatrix(m_img_vtk)

    return coil_actor


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


if __name__ == "__main__":
    # execute only if run as a script
    main()
