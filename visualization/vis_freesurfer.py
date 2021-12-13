#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xcoord - Tools for cross-software spatial coordinate manipulation
#
# This file is part of xcoord package which is released under copyright.
# See file LICENSE or go to website for full license details.
# Copyright (C) 2020 Victor Hugo Souza - All Rights Reserved
#
# Homepage: https://github.com/vhosouza/xcoord
# Contact: victor.souza@aalto.fi
# License: MIT License
#
# Authors: Victor Hugo Souza
# Date/version: 2.3.2020


import nibabel.freesurfer.io as fsio
import os
import vtk


def main():

    # define specific freesurfer polydata surface to read
    hemi = 'lh'  # lh or rh for left or right hemispheres
    surf = 'inflated'  # all surfaces exported by freesurfer, e.g., pial, sphere, smoothwm, curv, inflated...
    subj = 5

    fs_file = '{}.{}'.format(hemi, surf)
    fs_dir = os.environ['OneDrive'] + r'\data\nexstim_coord\freesurfer\ppM1_S{}\surf'.format(subj)
    fs_path = os.path.join(fs_dir, fs_file)

    vertices, faces, volume_info = fsio.read_geometry(fs_path, read_metadata=True)

    # create the methods to convert the nifti points to vtk object
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    colors = vtk.vtkNamedColors()

    # load the point, cell, and data attributes
    for n, xi in enumerate(vertices):
        points.InsertPoint(n, xi)
        scalars.InsertTuple1(n, n)
    for fc in faces:
        polys.InsertNextCell(make_vtk_id_list(fc))

    # assign the pieces to the vtkPolyData
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    polydata.GetPointData().SetScalars(scalars)

    # visualize
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(polydata.GetScalarRange())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(colors.GetColor3d("Cornsilk"))

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)

    # create the window interactor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    iren.Initialize()
    ren_win.Render()
    iren.Start()

    close_window(iren)


def make_vtk_id_list(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: a python iterable.
    :return: a vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def close_window(iren):
    """
    Close the vtk window and clean the allocated memory
    Not strictly needed for this small application but might be useful to larger ones

    :param iren: a vtkRenderWindowInteractor instance
    """
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()
    del render_window, iren


if __name__ == '__main__':
    main()


# using this method is a lot simpler but the code crashes right after pd.SetPolys
# from a quick search, it seems that for vtk is missing the proper ids of the faces and points,
# which is corrected by the current working method
#
# import numpy as np
# from vtk.util import numpy_support
#
# vtkvert = numpy_support.numpy_to_vtk(vertices)
# id_triangles = numpy_support.numpy_to_vtkIdTypeArray(faces.astype(np.int64))
#
# points = vtk.vtkPoints()
# points.SetData(vtkvert)
#
# triangles = vtk.vtkCellArray()
# triangles.SetCells(faces.shape[0], id_triangles)
#
# pd = vtk.vtkPolyData()
# pd.SetPoints(points)
# pd.SetPolys(triangles)