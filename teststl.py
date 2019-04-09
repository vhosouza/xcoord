#!/usr/bin/env python

import vtk

filename = "s01_gm.stl"
filename2 = "s01_skin.stl"
 
reader = vtk.vtkSTLReader()
reader.SetFileName(filename)

reader2 = vtk.vtkSTLReader()
reader2.SetFileName(filename2)
 
mapper = vtk.vtkPolyDataMapper()
mapper2 = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(reader.GetOutput())
    mapper2.SetInput(reader2.GetOutput())
else:
    mapper.SetInputConnection(reader.GetOutputPort())
    mapper2.SetInputConnection(reader2.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0., 0.)

actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)
actor2.GetProperty().SetOpacity(0.5)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
 
# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)
ren.AddActor(actor2)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()
