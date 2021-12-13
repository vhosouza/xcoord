import nibabel
import vtk
import numpy as np
from scipy import ndimage
import os
import matplotlib.pyplot as plt


def vtk_iso(vol, iso_thresh=1):
    # im_data = vol.tostring()
    # img = vtk.vtkImageImport()
    # img.CopyImportVoidPointer(im_data, len(im_data))
    # img.SetDataScalarType(vtk.VTK_UNSIGNED_SHORT)
    # img.SetNumberOfScalarComponents(1)
    # # img.SetDataExtent(0, vol.shape[2]-1, 0, vol.shape[1]-1, 0, vol.shape[0]-1)
    # # img.SetWholeExtent(0, vol.shape[2]-1, 0, vol.shape[1]-1, 0, vol.shape[0]-1)
    # img.SetDataExtent(0, vol.shape[2] - 1, 0, vol.shape[1] - 1, 0, vol.shape[0] - 1)
    # img.SetWholeExtent(0, vol.shape[2] - 1, 0, vol.shape[1] - 1, 0, vol.shape[0] - 1)
    iso = vtk.vtkMarchingCubes()
    iso.SetInputConnection(vol.GetOutputPort())
    iso.SetValue(0, iso_thresh)
    return iso


def vtk_smooth(iso, iter=20, relax=0.5, decimate=0.0):
    isoSmooth = vtk.vtkSmoothPolyDataFilter()
    if decimate > 0:
        deci = vtk.vtkDecimatePro()
        deci.SetInputConnection(iso.GetOutputPort())
        deci.SetTargetReduction(decimate)
        deci.PreserveTopologyOn()
        isoSmooth.SetInputConnection(deci.GetOutputPort())
    else:
        isoSmooth.SetInputConnection(iso.GetOutputPort())
    isoSmooth.SetNumberOfIterations(100)
    isoSmooth.BoundarySmoothingOn()
    isoSmooth.FeatureEdgeSmoothingOff()
    isoSmooth.SetFeatureAngle(45)
    isoSmooth.SetEdgeAngle(15)
    isoSmooth.SetRelaxationFactor(relax)
    return isoSmooth


def vtk_render_window(ren, iso, img=None, color=[0.5, 0.5, 0.5]):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(iso.GetOutputPort())
    normals.FlipNormalsOn()
    isoMapper = vtk.vtkPolyDataMapper()
    isoMapper.SetInputConnection(normals.GetOutputPort())
    isoMapper.ScalarVisibilityOff()

    isoActor = vtk.vtkActor()
    isoActor.SetMapper(isoMapper)
    isoActor.GetProperty().SetColor(color)

    # Add the actors to the renderer, set the background and size
    if img != None:
        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(img.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)
        outlineActor.VisibilityOff()
        ren.AddActor(outlineActor)

    ren.AddActor(isoActor)
    return ren


def montage(vol, ncols=None):
    """Returns a 2d image montage given a 3d volume."""
    ncols = ncols if ncols else int(np.ceil(np.sqrt(vol.shape[2])))
    rows = np.array_split(vol, range(ncols, vol.shape[2], ncols), axis=2)
    # ensure the last row is the same size as the others
    rows[-1] = np.dstack((rows[-1], np.zeros(rows[-1].shape[0:2] + (rows[0].shape[2]-rows[-1].shape[2],))))
    im = np.vstack([np.squeeze(np.hstack(np.dsplit(row, ncols))) for row in rows])
    return im


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


id_smooth = True
id_save = False
SHOW_AXES = True

# get the pial surface
data_dir = b'C:\Users\deoliv1\OneDrive - Aalto University\data\dti_navigation\juuso'
brainmask_path = os.path.join(data_dir, b'c2sub-P0_T1w_biascorrected.nii')
# # brainmask_path = os.path.join(data_dir, b'c1sub-P0_T1w_biascorrected.nii')
# brainmask = nibabel.load(brainmask_path.decode('utf-8'))
# brainmask_vol = brainmask.get_data()
# pial_iso, pial_img = vtk_iso(brainmask_vol)
#
# t1_path = os.path.join(data_dir, b'sub-P0_T1w_biascorrected.nii')
# # get the white matter surface
# t1 = nibabel.load(t1_path.decode('utf-8'))
# t1_vol = t1.get_data()
# # apply the brain mask to remove the scalp
# t1_vol[brainmask_vol == 0] = 0
# # segment the white matter
# print("Min: %d, Max: %d" % (np.min(t1_vol), np.max(t1_vol)))
# wm_vol = (t1_vol > 0).astype(np.float)
# wm_vol = ndimage.gaussian_filter(wm_vol.astype(float), 0.3)
# wm_vol = ndimage.measurements.label(wm_vol)[0]
# wm_vol = (wm_vol == 1).astype(np.uint16)
#
# im_mon = montage(t1_vol[:, :, 20:200:2])
# plt.imshow(im_mon, cmap='gray')
# plt.show()

T1_reader = vtk.vtkNIFTIImageReader()
T1_reader.SetFileName(brainmask_path)
T1_reader.Update()

refImage = T1_reader.GetOutputPort()
qFormMatrix = T1_reader.GetQFormMatrix()

refImageSpace2_xyz_transform = vtk.vtkTransform()
refImageSpace2_xyz_transform.SetMatrix(qFormMatrix)

print(qFormMatrix)

# iso = vtk.vtkMarchingCubes()
# iso.SetInputConnection(refImage)
# iso.SetValue(0, 1)
# iso.Update()

iso = vtk.vtkContourFilter()
iso.SetInputConnection(refImage)
iso.SetValue(0, 1)
iso.Update()

refImageSpace2_xyz = vtk.vtkTransformPolyDataFilter()
refImageSpace2_xyz.SetTransform(refImageSpace2_xyz_transform)
refImageSpace2_xyz.SetInputConnection(iso.GetOutputPort())
refImageSpace2_xyz.Update()

# wm_iso = vtk_iso(wm_vol)

if id_smooth:
    refImageSpace2_xyz = vtk_smooth(refImageSpace2_xyz, 20, 0.1, 0.2)
    # stl_path = os.path.join(data_dir, b'gm_orig_smooth.stl')
    stl_path = os.path.join(data_dir, b'wm_orig_smooth_world.stl')
else:
    # stl_path = os.path.join(data_dir, b'gm_orig.stl')
    stl_path = os.path.join(data_dir, b'wm_orig_world.stl')

if id_save:
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(stl_path)
    stlWriter.SetInputConnection(refImageSpace2_xyz.GetOutputPort())
    stlWriter.Write()

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

ren = vtk_render_window(ren, refImageSpace2_xyz)
ren = vtk_render_window(ren, iso)

if SHOW_AXES:
    add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
    add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
    add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

ren.SetBackground(1.0, 1.0, 1.0)
renWin.SetSize(450, 450)
## ren.GetActiveCamera().Elevation(235)
## ren.GetActiveCamera().SetViewUp(0,.5,-1)
## ren.GetActiveCamera().Azimuth(90)
iren.Initialize()
iren.Start()
