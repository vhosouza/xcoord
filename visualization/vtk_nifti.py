#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import vtk
import nibabel as nib

data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\baran\anat_reg_improve_20200609'
data_dir = data_dir.encode('utf-8')

nii_path = b'Baran_T1_inFODspace.nii'
act_path = os.path.join(data_dir, nii_path)

img = nib.load(act_path.decode('utf-8'))
# img = nib.as_closest_canonical(img)
# img.update_header()

img_data = img.get_data()
img_data_shape = img_data.shape

data_string = img_data.tostring()
dataImporter = vtk.vtkImageImport()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))
# dataImporter.SetDataScalarTypeToShort()
dataImporter.SetDataScalarType(vtk.VTK_UNSIGNED_SHORT)
dataImporter.SetNumberOfScalarComponents(1)
dataImporter.SetDataExtent(0, img_data_shape[2] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[0] - 1)
dataImporter.SetWholeExtent(0, img_data_shape[2] - 1, 0, img_data_shape[1] - 1, 0, img_data_shape[0] - 1)
dataImporter.Update()

mask_reader = vtk.vtkNIFTIImageReader()
mask_reader.SetFileName(act_path.decode('utf-8'))
mask_reader.Update()

temp_data = mask_reader.GetOutput()
new_data = vtk.vtkImageData()
new_data.DeepCopy(temp_data)

#outline
outline=vtk.vtkOutlineFilter()
outline.SetInputData(new_data)
outlineMapper=vtk.vtkPolyDataMapper()
outlineMapper.SetInputConnection(outline.GetOutputPort())
outlineActor = vtk.vtkActor()
outlineActor.SetMapper(outlineMapper)

#Picker
picker = vtk.vtkCellPicker()
picker.SetTolerance(0.005)

#PlaneWidget
planeWidgetX = vtk.vtkImagePlaneWidget()
planeWidgetX.DisplayTextOn()
planeWidgetX.SetInputData(new_data)
planeWidgetX.SetPlaneOrientationToXAxes()
planeWidgetX.SetSliceIndex(100)
planeWidgetX.SetPicker(picker)
planeWidgetX.SetKeyPressActivationValue("x")
prop1 = planeWidgetX.GetPlaneProperty()
prop1.SetColor(1, 0, 0)

planeWidgetY = vtk.vtkImagePlaneWidget()
planeWidgetY.DisplayTextOn()
planeWidgetY.SetInputData(new_data)
planeWidgetY.SetPlaneOrientationToYAxes()
planeWidgetY.SetSliceIndex(100)
planeWidgetY.SetPicker(picker)
planeWidgetY.SetKeyPressActivationValue("y")
prop2 = planeWidgetY.GetPlaneProperty()
prop2.SetColor(1, 1, 0)
planeWidgetY.SetLookupTable(planeWidgetX.GetLookupTable())

planeWidgetZ = vtk.vtkImagePlaneWidget()
planeWidgetZ.DisplayTextOn()
planeWidgetZ.SetInputData(new_data)
planeWidgetZ.SetPlaneOrientationToZAxes()
planeWidgetZ.SetSliceIndex(100)
planeWidgetZ.SetPicker(picker)
planeWidgetZ.SetKeyPressActivationValue("z")
prop2 = planeWidgetY.GetPlaneProperty()
prop2.SetColor(0, 0, 1)
planeWidgetZ.SetLookupTable(planeWidgetX.GetLookupTable())

#Renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 1)

#RenderWindow
renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)

#Add outlineactor
renderer.AddActor(outlineActor)
renwin.SetSize(800,800)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renwin)

#Load widget interactors and enable
planeWidgetX.SetInteractor(interactor)
planeWidgetX.On()
planeWidgetY.SetInteractor(interactor)
planeWidgetY.On()
planeWidgetZ.SetInteractor(interactor)
planeWidgetZ.On()

interactor.Initialize()
renwin.Render()
interactor.Start()