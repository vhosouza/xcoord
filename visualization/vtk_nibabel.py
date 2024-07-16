# -*- coding: utf-8 -*-

import vtk
import nibabel as nib
import os
import sys

# FOD_path = b"test_fod.nii"

if sys.platform == "win32":
    onedrive_path = os.environ.get('OneDrive')
elif (sys.platform == "darwin") or (sys.platform == "linux"):
    onedrive_path = os.path.expanduser('~/OneDrive - Aalto University')
else:
    onedrive_path = False
    print("Unsupported platform")

data_dir = os.path.join(onedrive_path, 'projects', 'nexstim', 'data', 'mri')
img_path = 'GJ_2008_anonym_t1_mpr_ns_sag_1_1_1_mm_20081021180940_2.nii.gz'
# data_dir = os.path.join(onedrive_path, 'data', 'dti_navigation', 'juuso')
# img_path = 'sub-P0_T1w_biascorrected.nii'
full_path = os.path.join(data_dir, img_path)
img = nib.load(full_path)

# data_dir = b'C:\Users\deoliv1\OneDrive - Aalto University\data\dti_navigation\juuso'
# img_path = b'sub-P0_T1w_biascorrected.nii'
# full_path = os.path.join(data_dir, img_path)
# img = nib.load(full_path.decode('utf-8'))

img_data = img.get_fdata()
data_string = img_data.tostring()

dataImporter = vtk.vtkImageImport()
dataImporter.SetDataScalarTypeToShort()
dataImporter.SetNumberOfScalarComponents(1)
dataImporter.CopyImportVoidPointer(data_string, len(data_string))
dataImporter.SetDataExtent(0, img_data.shape[2] - 1, 0, img_data.shape[1] - 1, 0, img_data.shape[0] - 1)
dataImporter.SetWholeExtent(0, img_data.shape[2] - 1, 0, img_data.shape[1] - 1, 0, img_data.shape[0] - 1)
dataImporter.Update()

new_data = vtk.vtkImageData()
new_data.DeepCopy(dataImporter.GetOutput())

#outline
outline = vtk.vtkOutlineFilter()
outline.SetInputData(new_data)
outlineMapper = vtk.vtkPolyDataMapper()
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
planeWidgetX.SetSliceIndex(50)
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
planeWidgetZ.SetSliceIndex(50)
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
renwin.SetSize(800, 800)

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
