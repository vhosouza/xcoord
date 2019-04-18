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

import os

import nibabel as nb
from nibabel.affines import apply_affine
import numpy as np
from scipy import io
import vtk

import load_coords as lc
import nexstim2mri as n2m


def main():

    SHOW_AXES = True
    SHOW_SCENE_AXES = True
    SHOW_COIL_AXES = True
    SHOW_SKIN = True
    SHOW_BRAIN = True
    SHOW_COIL = True
    SHOW_MARKERS = True
    TRANSF_COIL = True
    SHOW_PLANE = True
    SELECT_LANDMARKS = 'scalp'  # 'all', 'mri' 'scalp'
    SAVE_ID = False

    reorder = [0, 2, 1]
    flipx = [True, False, False]

    # reorder = [0, 1, 2]
    # flipx = [False, False, False]

    # default folder and subject
    # subj = 's03'
    subj = 'S5'
    id_extra = False  # 8, 9, 10, 12, False
    # data_dir = os.environ['OneDriveConsumer'] + '\\data\\nexstim_coord\\'
    data_dir = 'P:\\tms_eeg\\mTMS\\projects\\lateral ppTMS M1\\E-fields\\'
    # data_subj = data_dir + subj + '\\'
    simnibs_dir = data_dir + 'simnibs\\m2m_ppM1_%s_nc\\' % subj
    if id_extra:
        nav_dir = data_dir + 'nav_coordinates\\ppM1_%s_%d\\' % (subj, id_extra)
    else:
        nav_dir = data_dir + 'nav_coordinates\\ppM1_%s\\' % subj

    # filenames
    # coil_file = data_dir + 'magstim_fig8_coil.stl'
    coil_file = os.environ['OneDriveConsumer'] + '\\data\\nexstim_coord\\magstim_fig8_coil.stl'
    if id_extra:
        coord_file = nav_dir + 'ppM1_eximia_%s_%d.txt' % (subj, id_extra)
    else:
        coord_file = nav_dir + 'ppM1_eximia_%s.txt' % subj
    # img_file = data_subj + subj + '.nii'
    img_file = data_dir + 'mri\\ppM1_%s\\ppM1_%s.nii' % (subj, subj)
    brain_file = simnibs_dir + "wm.stl"
    skin_file = simnibs_dir + "skin.stl"
    if id_extra:
        output_file = nav_dir + 'transf_mat_%s_%d' % (subj, id_extra)
    else:
        output_file = nav_dir + 'transf_mat_%s' % subj

    coords = lc.load_nexstim(coord_file)
    # red, green, blue, maroon (dark red),
    # olive (shitty green), teal (petrol blue), yellow, orange
    col = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
           [.5, .5, 0.], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

    # extract image header shape and affine transformation from original nifti file
    imagedata = nb.squeeze_image(nb.load(img_file))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    img_shape = imagedata.header.get_data_shape()
    affine = imagedata.affine
    affine_I = np.identity(4)

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    if SELECT_LANDMARKS == 'mri':
        # MRI landmarks
        coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Coil Loc'], ['EF max']]
        pts_ref = [1, 2, 3, 7, 10]
    elif SELECT_LANDMARKS == 'all':
        # all coords
        coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Nose/Nasion'], ['Left ear'], ['Right ear'],
                     ['Coil Loc'], ['EF max']]
        pts_ref = [1, 2, 3, 5, 4, 6, 7, 10]
    elif SELECT_LANDMARKS == 'scalp':
        # scalp landmarks
        coord_mri = [['Nose/Nasion'], ['Left ear'], ['Right ear'], ['Coil Loc'], ['EF max']]
        pts_ref = [5, 4, 6, 7, 10]

    for n, pts_id in enumerate(pts_ref):
        coord_aux = n2m.coord_change(coords[pts_id][1:], img_shape, affine_I, flipx, reorder)
        [coord_mri[n].append(s) for s in coord_aux]

        if SHOW_MARKERS:
            marker_actor = add_marker(coord_aux, ren, col[n])

    print('\nOriginal coordinates from Nexstim: \n')
    [print(s) for s in coords]
    print('\nTransformed coordinates to MRI space: \n')
    [print(s) for s in coord_mri]

    # coil location, normal vector and direction vector
    coil_loc = coord_mri[-2][1:]
    coil_norm = coords[8][1:]
    coil_dir = coords[9][1:]

    # creating the coil coordinate system by adding a point in the direction of each given coil vector
    # the additional vector is just the cross product from coil direction and coil normal vectors
    # origin of the coordinate system is the coil location given by Nexstim
    # the vec_length is to allow line creation with visible length in VTK scene
    vec_length = 75
    p1 = coords[7][1:]
    p2 = [x - vec_length * y for x, y in zip(p1, coil_norm)]
    p2_norm = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    p2 = [x + vec_length * y for x, y in zip(p1, coil_dir)]
    p2_dir = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    coil_face = np.cross(coil_norm, coil_dir)
    p2 = [x + vec_length * y for x, y in zip(p1, coil_face.tolist())]
    p2_face = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    if SHOW_BRAIN:
        brain_actor = load_stl(brain_file, ren, colour=[0., 1., 1.], opacity=0.7, user_matrix=np.linalg.inv(affine))
    if SHOW_SKIN:
        skin_actor = load_stl(skin_file, ren, opacity=0.5, user_matrix=np.linalg.inv(affine))

    if SHOW_COIL:
        # Coil direction unit vector
        u1 = np.asarray(p2_dir) - np.asarray(coil_loc)
        u1_n = u1 / np.linalg.norm(u1)
        # Coil normal unit vector
        u2 = np.asarray(p2_norm) - np.asarray(coil_loc)
        u2_n = u2 / np.linalg.norm(u2)
        # Coil face unit vector
        u3 = np.asarray(p2_face) - np.asarray(coil_loc)
        u3_n = u3 / np.linalg.norm(u3)

        transf_matrix = np.identity(4)
        if TRANSF_COIL:
            transf_matrix[:3, 0] = u1_n
            transf_matrix[:3, 1] = u2_n
            transf_matrix[:3, 2] = u3_n
            transf_matrix[:3, 3] = coil_loc[:]

        if SAVE_ID:
            coord_dict = {'coil_orient': transf_matrix[:3, :3], 'coil_loc': coil_loc}
            io.savemat(output_file + '.mat', coord_dict)
            hdr_names = ';'.join(['m' + str(i) + str(j) for i in range(1, 5) for j in range(1, 5)])
            np.savetxt(output_file + '.txt', transf_matrix.reshape([1, 16]), delimiter=';', header=hdr_names)

        # reposition STL object prior to transformation matrix
        # [translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z]
        repos = [0., -6., 0., 0., -90., 90.]
        act_coil = load_stl(coil_file, ren, replace=repos, user_matrix=transf_matrix)

        # the absolute value of the determinant indicates the scaling factor
        # the sign of the determinant indicates how it affects the orientation: if positive maintain the
        # original orientation and if negative inverts all the orientations (flip the object inside-out)'
        # the negative determinant is what makes objects in VTK scene to become black
        print('Transformation matrix: \n', transf_matrix, '\n')
        print('Determinant: ', np.linalg.det(transf_matrix))

    if SHOW_PLANE:
        act_plane = add_plane(ren, user_matrix=transf_matrix)

    # Add axes to scene origin
    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Add axes to object origin
    if SHOW_COIL_AXES:
        add_line(ren, coil_loc, p2_norm, color=[.0, 1.0, 0.0])
        add_line(ren, coil_loc, p2_dir, color=[1.0, .0, .0])
        add_line(ren, coil_loc, p2_face, color=[.0, .0, 1.0])

    # Add interactive axes to scene
    if SHOW_SCENE_AXES:
        axes = vtk.vtkAxesActor()
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(iren)
        # widget.SetViewport(0.0, 0.0, 0.4, 0.4)
        widget.SetEnabled(1)
        widget.InteractiveOn()

    # Enable user interface interactor
    iren.Initialize()
    ren.ResetCamera()
    ren_win.Render()
    iren.Start()


def add_marker(coord, ren, color):
    # x, y, z = coord

    ball_ref = vtk.vtkSphereSource()
    ball_ref.SetRadius(2)
    ball_ref.SetCenter(coord)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ball_ref.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(color)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    ren.AddActor(actor)

    return actor


def add_plane(ren, coil_center=[0., 0., 0.], coil_normal=[0., 1., 0.], user_matrix=np.identity(4)):

    coil_plane = vtk.vtkPlaneSource()
    coil_plane.SetOrigin(0, 0, 0)
    coil_plane.SetPoint1(50, 0, 0)
    coil_plane.SetPoint2(0, 0, 100)
    coil_plane.SetCenter(coil_center)
    coil_plane.SetNormal(coil_normal)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(coil_plane.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(0.5, 0., 0.5)
    # prop.SetColor(1., 0., 0.)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    actor.SetUserMatrix(matrix_vtk)

    # Assign actor to the renderer
    ren.AddActor(actor)

    return actor


def load_stl(stl_path, ren, opacity=1., visibility=1, position=False, colour=False, replace=False, user_matrix=np.identity(4)):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

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
        transform_filt.SetInputConnection(reader.GetOutputPort())
        transform_filt.Update()

    mapper = vtk.vtkPolyDataMapper()

    if vtk.VTK_MAJOR_VERSION <= 5:
        if replace:
            mapper.SetInput(transform_filt.GetOutput())
        else:
            mapper.SetInput(reader.GetOutput())
    else:
        if replace:
            mapper.SetInputConnection(transform_filt.GetOutputPort())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.SetVisibility(visibility)

    if colour:
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
