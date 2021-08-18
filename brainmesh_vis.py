import vtk
import pyacvd
import os
import pyvista
import numpy as np


class Brain:
    def __init__(self, img_path, mask_path):
        self.peel = []
        self.peelActors = []

        T1_reader = vtk.vtkNIFTIImageReader()
        T1_reader.SetFileName(img_path)
        T1_reader.Update()

        # self.refImage = vtk.vtkImageData()
        self.refImage = T1_reader.GetOutput()

        mask_reader = vtk.vtkNIFTIImageReader()
        mask_reader.SetFileName(mask_path)
        mask_reader.Update()

        mc = vtk.vtkContourFilter()
        mc.SetInputConnection(mask_reader.GetOutputPort())
        mc.SetValue(0, 1)
        mc.Update()

        refSurface = vtk.vtkPolyData()
        refSurface = mc.GetOutput()

        tmpPeel = vtk.vtkPolyData()
        tmpPeel = downsample(refSurface)

        mask_sFormMatrix = vtk.vtkMatrix4x4()
        mask_sFormMatrix = mask_reader.GetSFormMatrix()

        mask_ijk2xyz = vtk.vtkTransform()
        mask_ijk2xyz.SetMatrix(mask_sFormMatrix)

        mask_ijk2xyz_filter = vtk.vtkTransformPolyDataFilter()
        mask_ijk2xyz_filter.SetInputData(tmpPeel)
        mask_ijk2xyz_filter.SetTransform(mask_ijk2xyz)
        mask_ijk2xyz_filter.Update()

        tmpPeel = smooth(mask_ijk2xyz_filter.GetOutput())
        tmpPeel = fixMesh(tmpPeel)
        tmpPeel = cleanMesh(tmpPeel)
        tmpPeel = upsample(tmpPeel)
        tmpPeel = smooth(tmpPeel)
        tmpPeel = fixMesh(tmpPeel)
        tmpPeel = cleanMesh(tmpPeel)

        # sFormMatrix = vtk.vtkMatrix4x4()
        qFormMatrix = T1_reader.GetQFormMatrix()
        # sFormMatrix = T1_reader.GetSFormMatrix()

        refImageSpace2_xyz_transform = vtk.vtkTransform()
        refImageSpace2_xyz_transform.SetMatrix(qFormMatrix)

        self.refImageSpace2_xyz = vtk.vtkTransformPolyDataFilter()
        self.refImageSpace2_xyz.SetTransform(refImageSpace2_xyz_transform)

        xyz2_refImageSpace_transform = vtk.vtkTransform()
        qFormMatrix.Invert()
        xyz2_refImageSpace_transform.SetMatrix(qFormMatrix)

        self.xyz2_refImageSpace = vtk.vtkTransformPolyDataFilter()
        self.xyz2_refImageSpace.SetTransform(xyz2_refImageSpace_transform)

        # self.currentPeel = vtk.vtkPolyData()
        self.currentPeel = tmpPeel
        self.currentPeelNo = 0
        self.mapImageOnCurrentPeel()

        newPeel = vtk.vtkPolyData()
        newPeel.DeepCopy(self.currentPeel)
        self.peel.append(newPeel)
        self.currentPeelActor = vtk.vtkActor()
        self.getCurrentPeelActor()
        self.peelActors.append(self.currentPeelActor)

        self.numberOfPeels = 15
        self.peelDown()

    def get_actor(self, n):
        return self.getPeelActor(n), self.peel[n]

    def sliceDown(self):
        # Warp using the normals
        warp = vtk.vtkWarpVector()
        warp.SetInputData(fixMesh(downsample(self.currentPeel)))  # fixMesh here updates normals needed for warping
        warp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
                                    vtk.vtkDataSetAttributes().NORMALS)
        warp.SetScaleFactor(-1)
        warp.Update()

        out = vtk.vtkPolyData()
        out = upsample(warp.GetPolyDataOutput())
        out = smooth(out)
        out = fixMesh(out)
        out = cleanMesh(out)

        self.currentPeel = out

    # def sliceUp(self):
    #     # Warp using the normals
    #     warp = vtk.vtkWarpVector()
    #     # warp.SetInputData(fixMesh(downsample(currentPeel))) # fixMesh here updates normals needed for warping
    #     warp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS,
    #                                 vtk.vtkDataSetAttributes().NORMALS)
    #     warp.SetScaleFactor(1)
    #     warp.Update()
    #
    #     out = vtk.vtkPolyData()
    #     out = upsample(warp.GetPolyDataOutput())
    #     out = smooth(out)
    #     out = fixMesh(out)
    #     out = cleanMesh(out)
    #
    #     currentPeel = out

    def mapImageOnCurrentPeel(self):
        self.xyz2_refImageSpace.SetInputData(self.currentPeel)
        self.xyz2_refImageSpace.Update()

        probe = vtk.vtkProbeFilter()
        probe.SetInputData(self.xyz2_refImageSpace.GetOutput())
        probe.SetSourceData(self.refImage)
        probe.Update()

        self.refImageSpace2_xyz.SetInputData(probe.GetOutput())
        self.refImageSpace2_xyz.Update()

        self.currentPeel = self.refImageSpace2_xyz.GetOutput()

    def peelDown(self):
        for i in range(0, self.numberOfPeels):
            self.sliceDown()
            self.mapImageOnCurrentPeel()

            newPeel = vtk.vtkPolyData()
            newPeel.DeepCopy(self.currentPeel)
            self.peel.append(newPeel)

            # getCurrentPeelActor()
            # newPeelActor = vtk.vtkActor()
            # newPeelActor = currentPeelActor
            # peelActors.push_back(newPeelActor)

            self.currentPeelNo += 1

    def getPeelActor(self, p):
        colors = vtk.vtkNamedColors()
        # Create the color map
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetNumberOfColors(512)
        colorLookupTable.SetSaturationRange(0, 0)
        colorLookupTable.SetHueRange(0, 0)
        colorLookupTable.SetValueRange(0, 1)
        # colorLookupTable.SetTableRange(0, 1000)
        # colorLookupTable.SetTableRange(0, 250)
        colorLookupTable.SetTableRange(0, 200)
        # colorLookupTable.SetTableRange(0, 150)
        colorLookupTable.Build()

        # Set mapper auto
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputData(self.peel[p])
        # mapper.SetScalarRange(0, 1000)
        # mapper.SetScalarRange(0, 250)
        mapper.SetScalarRange(0, 200)
        # mapper.SetScalarRange(0, 150)
        mapper.SetLookupTable(colorLookupTable)
        mapper.InterpolateScalarsBeforeMappingOn()

        # Set actor
        self.currentPeelActor.SetMapper(mapper)

        return self.currentPeelActor

    def getCurrentPeelActor(self):
        colors = vtk.vtkNamedColors()

        # Create the color map
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetNumberOfColors(512)
        colorLookupTable.SetSaturationRange(0, 0)
        colorLookupTable.SetHueRange(0, 0)
        colorLookupTable.SetValueRange(0, 1)
        # colorLookupTable.SetTableRange(0, 1000)
        # colorLookupTable.SetTableRange(0, 250)
        colorLookupTable.SetTableRange(0, 200)
        # colorLookupTable.SetTableRange(0, 150)
        colorLookupTable.Build()

        # Set mapper auto
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputData(self.currentPeel)
        # mapper.SetScalarRange(0, 1000)
        # mapper.SetScalarRange(0, 250)
        mapper.SetScalarRange(0, 200)
        # mapper.SetScalarRange(0, 150)
        mapper.SetLookupTable(colorLookupTable)
        mapper.InterpolateScalarsBeforeMappingOn()

        # Set actor
        self.currentPeelActor.SetMapper(mapper)
        self.currentPeelActor.GetProperty().SetBackfaceCulling(1)
        self.currentPeelActor.GetProperty().SetOpacity(0.5)

        return self.currentPeelActor


def cleanMesh(inp):
    cleaned = vtk.vtkCleanPolyData()
    cleaned.SetInputData(inp)
    cleaned.Update()

    return cleaned.GetOutput()


def fixMesh(inp):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(inp)
    normals.SetFeatureAngle(160)
    normals.SplittingOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    return normals.GetOutput()


def upsample(inp):
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(inp)
    triangles.Update()

    subdivisionFilter = vtk.vtkLinearSubdivisionFilter()
    subdivisionFilter.SetInputData(triangles.GetOutput())
    subdivisionFilter.SetNumberOfSubdivisions(2)
    subdivisionFilter.Update()

    return subdivisionFilter.GetOutput()


def smooth(inp):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(inp)
    smoother.SetNumberOfIterations(20)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(175)
    smoother.SetPassBand(0.1)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother.GetOutput()


def downsample(inp):
    # surface = vtk.vtkSurface()
    # surface.CreateFromPolyData(inp)
    #
    # areas = vtk.vtkDoubleArray()
    # areas = surface.GetTrianglesAreas()
    # surfaceArea = 0
    #
    # for i in range(0, areas.GetSize()):
    #     surfaceArea += areas.GetValue(i)
    #
    # clusterNumber = surfaceArea / 20

    mesh = pyvista.PolyData(inp)

    # Create clustering object
    clus = pyacvd.Clustering(mesh)
    # mesh is not dense enough for uniform remeshing
    # clus.subdivide(3)
    clus.cluster(3000)
    Remesh = clus.create_mesh()

    # print(Remesh)

    # Remesh = vtk.vtkIsotropicDiscreteRemeshing()
    # Remesh.SetInput(surface)
    # Remesh.SetFileLoadSaveOption(0)
    # Remesh.SetNumberOfClusters(clusterNumber)
    # Remesh.SetConsoleOutput(0)
    # Remesh.GetMetric().SetGradation(0)
    # Remesh.SetDisplay(0)
    # Remesh.Remesh()

    # out = vtk.vtkPolyData()
    # out.SetPoints(Remesh.GetOutput().GetPoints())
    # out.SetPolys(Remesh.GetOutput().GetPolys())

    return Remesh


# def brain_downsample(inp):
#
#     surface = vtk.vtkSurface()
#     surface.CreateFromPolyData(inp)
#
#     areas = vtk.vtkDoubleArray()
#     areas = surface.GetTrianglesAreas()
#
#     surfaceArea = 0
#
#     for i in range(0, areas.GetSize()):
#         surfaceArea += areas.GetValue(i)
#
#     clusterNumber = surfaceArea / 2
#
#     Remesh = vtk.vtkIsotropicDiscreteRemeshing()
#     Remesh.SetInput(surface)
#     Remesh.SetFileLoadSaveOption(0)
#     Remesh.SetNumberOfClusters(clusterNumber)
#     Remesh.SetConsoleOutput(0)
#     Remesh.GetMetric().SetGradation(0)
#     Remesh.SetDisplay(0)
#     Remesh.Remesh()
#
#     out = vtk.vtkPolyData()
#     out.SetPoints(Remesh.GetOutput().GetPoints())
#     out.SetPolys(Remesh.GetOutput().GetPolys())
#
#     return out


def readBrain(brain_fname):
    fname = str(brain_fname)

    reader = vtk.vtkXMLPolyDataReader()
    # auto
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname.c_str())
    reader.Update()

    surface = vtk.vtkPolyData()
    surface = reader.GetOutput()
    # surface = brain_downsample(surface)

    # Write the file
    writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetFileName("100307_right_GM.vtp")
    # writer.SetInputData(surface)
    # writer.Write()

    return surface


def drawBrain(brain):
    colors = vtk.vtkNamedColors()

    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputData(brain)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetColor(colors.GetColor4d("Pink").GetData())
    actor.GetProperty().SetOpacity(0.25)
    actor.GetProperty().SetShading(1)
    actor.GetProperty().SetSpecular(1)
    actor.GetProperty().SetSpecularPower(100)
    actor.GetProperty().SetAmbient(0)
    actor.GetProperty().SetDiffuse(1)
    actor.GetProperty().SetSpecularColor(colors.GetColor4d("White").GetData())
    actor.GetProperty().SetInterpolationToGouraud()
    actor.GetProperty().SetBackfaceCulling(1)

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


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True

    # data_dir = b'C:\Users\img_folder'  #PUT YOUR IMAGE DIRECTORY HERE
    data_dir = b'C:\Users\deoliv1\OneDrive - Aalto University\data\dti_navigation\joonas'

    mask_file = b'dMRI_mask_T1_space_fixed.nii'
    mask_path = os.path.join(data_dir, mask_file)

    img_file = b'sub-S1_ses-S8741_T1w.nii'
    img_path = os.path.join(data_dir, img_file)

    # imagedata = nb.squeeze_image(nb.load(img_path.decode('utf-8')))
    # imagedata = nb.as_closest_canonical(imagedata)
    # imagedata.update_header()
    # pix_dim = imagedata.header.get_zooms()
    # img_shape = imagedata.header.get_data_shape()
    #
    # print("pix_dim: {0}, img_shape: {0}".format(pix_dim, img_shape))
    #
    # if AFFINE_IMG:
    #     affine = imagedata.affine
    #     if NO_SCALE:
    #         scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
    #         affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)
    # else:
    #     affine = np.identity(4)
    #
    # print("affine: {0}\n".format(affine))

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Assign actor to the renderer
    brain_actor, peel = Brain(img_path, mask_path).get_actor(14)

    ren.AddActor(brain_actor)

    # Enable user interface interactor
    iren.Initialize()
    ren_win.Render()
    iren.Start()


if __name__ == '__main__':
    main()
