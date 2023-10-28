#!/usr/bin/env python

"""
This is (almost) a direct C++ to Python transliteration of
 <VTK-root>/Examples/DataManipulation/Cxx/Cube.cxx from the VTK
 source distribution, which "shows how to manually create vtkPolyData"

A convenience function, mkVtkIdList(), has been added and one if/else
 so the example also works in version 6 or later.
If your VTK version is 5.x then remove the line: colors = vtk.vtkNamedColors()
 and replace the set background parameters with (1.0, 0.9688, 0.8594)

"""

import os
import vtk
import nibabel.freesurfer.io as fsio


def mkVtkIdList(it):
    """
    Makes a vtkIdList from a Python iterable. I'm kinda surprised that
     this is necessary, since I assumed that this kind of thing would
     have been built into the wrapper and happen transparently, but it
     seems not.

    :param it: A python iterable.
    :return: A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def main():

    save_id = False
    subj = 5
    fs_file = 'lh.pial'
    fs_dir = os.environ['OneDrive'] + r'\data\nexstim_coord\freesurfer\ppM1_S{}\surf'.format(subj)
    fs_path = os.path.join(fs_dir, fs_file)
    filename_save = 'lh.pial_8.stl'

    x, pts, volume_info = fsio.read_geometry(fs_path, read_metadata=True)

    # We'll create the building blocks of polydata including data attributes.
    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    colors = vtk.vtkNamedColors()

    # Load the point, cell, and data attributes.
    for i, xi in enumerate(x):
        points.InsertPoint(i, xi)
    for pt in pts:
        polys.InsertNextCell(mkVtkIdList(pt))
    for i, _ in enumerate(x):
        scalars.InsertTuple1(i, i)

    # We now assign the pieces to the vtkPolyData.
    cube.SetPoints(points)
    cube.SetPolys(polys)
    cube.GetPointData().SetScalars(scalars)

    if save_id:
        # Write the stl file to disk
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileName(filename_save)
        stlWriter.SetInputData(cube)
        stlWriter.Write()

    # Now we'll look at it.
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputData(cube)
    cubeMapper.SetScalarRange(cube.GetScalarRange())
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)

    # The usual rendering stuff.
    camera = vtk.vtkCamera()
    camera.SetPosition(1, 1, 1)
    camera.SetFocalPoint(0, 0, 0)

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renderer.AddActor(cubeActor)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.SetBackground(colors.GetColor3d("Cornsilk"))
    # renderer.SetBackground(1.0, 0.9688, 0.8594)

    renWin.SetSize(600, 600)

    # interact with data
    renWin.Render()
    iren.Start()


if __name__ == "__main__":
    main()
