#!/usr/bin/env python
# -*- coding: utf-8 -*-

import vtk
import numpy as np


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

    # outline
    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(transform_filt.GetOutputPort())
    # mapper_outline = vtk.vtkPolyDataMapper()
    # mapper_outline.SetInputConnection(outline.GetOutputPort())
    # actor_outline = vtk.vtkActor()
    # actor_outline.SetMapper(mapper_outline)

    if colour:
        if type(colour) is str:
            actor.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d(colour))
            actor.GetProperty().SetSpecular(.3)
            actor.GetProperty().SetSpecularPower(20)
            # actor_outline.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d("SkinColor"))
            # actor_outline.GetProperty().SetSpecular(.3)
            # actor_outline.GetProperty().SetSpecularPower(20)

        else:
            actor.GetProperty().SetColor(colour)
            # actor_outline.GetProperty().SetColor(colour)

    if position:
        actor.SetPosition(position)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    actor.SetUserMatrix(matrix_vtk)
    # actor_outline.SetUserMatrix(matrix_vtk)

    # Assign actor to the renderer
    ren.AddActor(actor)
    # ren.AddActor(actor_outline)

    return actor


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

    return actor


def add_marker(coord, ren, color, radius, opacity=1.):
    ball_ref = vtk.vtkSphereSource()
    ball_ref.SetRadius(radius)
    ball_ref.SetCenter(coord)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ball_ref.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(color)
    prop.SetOpacity(opacity)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    ren.AddActor(actor)

    return actor


def create_window(size=(800, 800)):
    ren = vtk.vtkRenderer()
    ren.SetUseDepthPeeling(1)
    ren.SetOcclusionRatio(0.1)
    ren.SetMaximumNumberOfPeels(100)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(*size)
    ren_win.SetMultiSamples(0)
    ren_win.SetAlphaBitPlanes(1)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    return ren, ren_win, iren
