"""
Basic Visualization
===================
Visualize freesurface surface in a VTK scene
"""

import nibabel.freesurfer.io as fsio
import vtk
from vtk.util import numpy_support
import numpy as np
#from stl import mesh

subject_id = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

fil = r'C:\Users\victo\OneDrive\data\nexstim_coord\freesurfer\ppM1_S1\surf\lh.pial'

vertices, faces, volume_info = fsio.read_geometry(fil, read_metadata=True)

vtkvert = numpy_support.numpy_to_vtk(vertices)
id_triangles = numpy_support.numpy_to_vtkIdTypeArray(faces.astype(np.int64))

points = vtk.vtkPoints()
points.SetData(vtkvert)

triangles = vtk.vtkCellArray()
triangles.SetCells(faces.shape[0], id_triangles)

pd = vtk.vtkPolyData()
pd.SetPoints(points)
pd.SetPolys(triangles)

# Visualize
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(pd)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

ren = vtk.vtkRenderer()
ren.ResetCamera()
ren.AddActor(actor)

ren_win = vtk.vtkRenderWindow()
ren_win.AddRenderer(ren)
ren_win.SetSize(800, 800)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)

ren.ResetCameraClippingRange()

iren.Initialize()
iren.Start()
