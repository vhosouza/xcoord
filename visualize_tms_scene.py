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
import numpy as np
from scipy import io
import transformations as tf
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
    SHOW_PLANE = False
    SELECT_LANDMARKS = 'scalp'  # 'all', 'mri' 'scalp'
    SAVE_ID = False
    AFFINE_IMG = True
    NO_SCALE = True
    SCREENSHOT = False

    reorder = [0, 2, 1]
    flipx = [True, False, False]

    # reorder = [0, 1, 2]
    # flipx = [False, False, False]

    # default folder and subject
    subj = 'S5'  # 1 - 9
    id_extra = False  # 8, 9, 10, 12, False
    data_dir = r'P:\tms_eeg\mTMS\projects\2016 Lateral ppTMS M1\E-fields'
    simnibs_dir = os.path.join(data_dir, 'simnibs', 'm2m_ppM1_{}_nc'.format(subj))

    if id_extra and subj == 'S1':
        subj_id = '{}_{}'.format(subj, id_extra)
    else:
        subj_id = '{}'.format(subj)

    nav_dir = os.path.join(data_dir, 'nav_coordinates', 'ppM1_{}'.format(subj_id))

    # filenames
    coil_file = os.path.join(os.environ['OneDrive'], 'data', 'nexstim_coord', 'magstim_fig8_coil.stl')

    coord_file = os.path.join(nav_dir, 'ppM1_eximia_{}.txt'.format(subj_id))

    img_file = os.path.join(data_dir, r'mri\ppM1_{}\ppM1_{}.nii'.format(subj, subj))
    brain_file = os.path.join(simnibs_dir, 'wm.stl')
    skin_file = os.path.join(simnibs_dir, 'skin.stl')

    output_file = os.path.join(nav_dir, 'transf_mat_{}'.format(subj_id))

    coords = lc.load_nexstim(coord_file)
    # red, green, blue, maroon (dark red),
    # olive (shitty green), teal (petrol blue), yellow, orange
    col = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
           [.5, .5, 0.], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

    # extract image header shape and affine transformation from original nifti file
    imagedata = nb.squeeze_image(nb.load(img_file))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    print("Pixel size: \n")
    print(pix_dim)
    print("\nImage shape: \n")
    print(img_shape)

    affine_aux = imagedata.affine.copy()
    if NO_SCALE:
        scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
        affine_aux = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)

    if AFFINE_IMG:
        affine = affine_aux
        # if NO_SCALE:
        #     scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
        #     affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)
    else:
        affine = np.identity(4)
    # affine_I = np.identity(4)

    # create a camera, render window and renderer
    camera = vtk.vtkCamera()
    camera.SetPosition(0, 1000, 0)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    camera.ComputeViewPlaneNormal()
    camera.Azimuth(90.0)
    camera.Elevation(10.0)

    ren = vtk.vtkRenderer()
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    camera.Dolly(1.5)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)

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
        hdr_mri = ['Nose/Nasion', 'Left ear', 'Right ear', 'Coil Loc', 'EF max']
        pts_ref = [5, 4, 6, 7, 10]

    coords_np = np.zeros([len(pts_ref), 3])

    for n, pts_id in enumerate(pts_ref):
        # to keep in the MRI space use the identity as the affine
        # coord_aux = n2m.coord_change(coords[pts_id][1:], img_shape, affine_I, flipx, reorder)
        # affine_trans = affine_I.copy()
        # affine_trans = affine.copy()
        # affine_trans[:3, -1] = affine[:3, -1]
        coord_aux = n2m.coord_change(coords[pts_id][1:], img_shape, affine, flipx, reorder)
        coords_np[n, :] = coord_aux
        [coord_mri[n].append(s) for s in coord_aux]

        if SHOW_MARKERS:
            marker_actor = add_marker(coord_aux, ren, col[n])

    if id_extra:
        # compare coil locations in experiments with 8, 9, 10 and 12 mm shifts
        # MRI Nexstim space: 8, 9, 10, 12 mm coil locations
        # coord_others = [[122.2, 198.8, 99.7],
        #                 [121.1, 200.4, 100.1],
        #                 [120.5, 200.7, 98.2],
        #                 [117.7, 202.9, 96.6]]
        if AFFINE_IMG:
            # World space: 8, 9, 10, 12 mm coil locations
            coord_others = [[-42.60270233154297, 28.266497802734378, 81.02450256347657],
                            [-41.50270233154296, 28.66649780273437, 82.62450256347657],
                            [-40.90270233154297, 26.766497802734378, 82.92450256347655],
                            [-38.10270233154297, 25.16649780273437, 85.12450256347657]]
        else:
            # MRI space reordered and flipped: 8, 9, 10, 12 mm coil locations
            coord_others = [[27.8,  99.7, 198.8],
                            [28.9, 100.1, 200.4],
                            [29.5, 98.2, 200.7],
                            [32.3,  96.6, 202.9]]

        col_others = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]
        for n, c in enumerate(coord_others):
            marker_actor = add_marker(c, ren, col_others[n])

    print('\nOriginal coordinates from Nexstim: \n')
    [print(s) for s in coords]
    print('\nMRI coordinates flipped and reordered: \n')
    [print(s) for s in coords_np]
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
    p2 = [x + vec_length * y for x, y in zip(p1, coil_norm)]
    p2_norm = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    p2 = [x + vec_length * y for x, y in zip(p1, coil_dir)]
    p2_dir = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    coil_face = np.cross(coil_norm, coil_dir)
    p2 = [x - vec_length * y for x, y in zip(p1, coil_face.tolist())]
    p2_face = n2m.coord_change(p2, img_shape, affine, flipx, reorder)

    # Coil face unit vector (X)
    u1 = np.asarray(p2_face) - np.asarray(coil_loc)
    u1_n = u1 / np.linalg.norm(u1)
    # Coil direction unit vector (Y)
    u2 = np.asarray(p2_dir) - np.asarray(coil_loc)
    u2_n = u2 / np.linalg.norm(u2)
    # Coil normal unit vector (Z)
    u3 = np.asarray(p2_norm) - np.asarray(coil_loc)
    u3_n = u3 / np.linalg.norm(u3)

    transf_matrix = np.identity(4)
    if TRANSF_COIL:
        transf_matrix[:3, 0] = u1_n
        transf_matrix[:3, 1] = u2_n
        transf_matrix[:3, 2] = u3_n
        transf_matrix[:3, 3] = coil_loc[:]

    # the absolute value of the determinant indicates the scaling factor
    # the sign of the determinant indicates how it affects the orientation: if positive maintain the
    # original orientation and if negative inverts all the orientations (flip the object inside-out)'
    # the negative determinant is what makes objects in VTK scene to become black
    print('Transformation matrix: \n', transf_matrix, '\n')
    print('Determinant: ', np.linalg.det(transf_matrix))

    if SAVE_ID:
        coord_dict = {'m_affine': transf_matrix, 'coords_labels': hdr_mri, 'coords': coords_np}
        io.savemat(output_file + '.mat', coord_dict)
        hdr_names = ';'.join(['m' + str(i) + str(j) for i in range(1, 5) for j in range(1, 5)])
        np.savetxt(output_file + '.txt', transf_matrix.reshape([1, 16]), delimiter=';', header=hdr_names)

    if SHOW_BRAIN:
        if AFFINE_IMG:
            brain_actor = load_stl(brain_file, ren, colour=[0., 1., 1.], opacity=1.)
        else:
            # to visualize brain in MRI space
            brain_actor = load_stl(brain_file, ren, colour=[0., 1., 1.], opacity=1., user_matrix=np.linalg.inv(affine_aux))
    if SHOW_SKIN:
        if AFFINE_IMG:
            skin_actor = load_stl(skin_file, ren, colour="SkinColor", opacity=.4)
        else:
            # to visualize skin in MRI space
            skin_actor = load_stl(skin_file, ren, colour="SkinColor", opacity=.4, user_matrix=np.linalg.inv(affine_aux))

    if SHOW_COIL:
        # reposition STL object prior to transformation matrix
        # [translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z]
        # old translation when using Y as normal vector
        # repos = [0., -6., 0., 0., -90., 90.]
        # Translate coil loc coordinate to coil bottom
        # repos = [0., 0., 5.5, 0., 0., 180.]
        repos = [0., 0., 0., 0., 0., 180.]
        # SimNIBS coil orientation requires 180deg around Y
        # Ry = tf.rotation_matrix(np.pi, [0, 1, 0], [0, 0, 0])
        # transf_matrix = transf_matrix @ Ry
        act_coil = load_stl(coil_file, ren, replace=repos, user_matrix=transf_matrix, opacity=.3)

    if SHOW_PLANE:
        act_plane = add_plane(ren, user_matrix=transf_matrix)

    # Add axes to scene origin
    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Add axes to object origin
    if SHOW_COIL_AXES:
        add_line(ren, coil_loc, p2_norm, color=[.0, .0, 1.0])
        add_line(ren, coil_loc, p2_dir, color=[.0, 1.0, .0])
        add_line(ren, coil_loc, p2_face, color=[1.0, .0, .0])

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

    if SCREENSHOT:
        # screenshot of VTK scene
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(ren_win)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("screenshot.png")
        writer.SetInput(w2if.GetOutput())
        writer.Write()

    # Enable user interface interactor
    # ren_win.Render()

    ren.ResetCameraClippingRange()

    iren.Initialize()
    iren.Start()


def add_marker(coord, ren, color):
    # x, y, z = coord

    ball_ref = vtk.vtkSphereSource()
    ball_ref.SetRadius(1)
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


def add_plane(ren, coil_center=[0., 0., 0.], coil_normal=[0., 0., 1.], user_matrix=np.identity(4)):

    coil_plane = vtk.vtkPlaneSource()
    coil_plane.SetOrigin(0, 0, 0)
    coil_plane.SetPoint1(0, 50, 0)
    coil_plane.SetPoint2(100, 0, 0)
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
