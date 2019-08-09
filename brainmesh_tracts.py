import vtk
import pyacvd
import os
import pyvista
import numpy as np
import Trekker


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

        self.numberOfPeels = 2
        self.peelDown()

    def get_actor(self, n):
        return self.getPeelActor(n)

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


def visualizeTracks(renderer, renderWindow, tracker, seed, replace, user_matrix):
    # Input the seed to the tracker object
    tracker.set_seeds(seed)

    # Run the tracker
    # This step will create N tracks if seed is a 3xN matrix
    tractogram = tracker.run()

    # Convert the first track to a vtkActor, i.e., tractogram[0] is the track
    # computed for the first seed
    trkActor = trk2vtkActor(tractogram[0], replace)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    trkActor.SetUserMatrix(matrix_vtk)

    renderer.AddActor(trkActor)
    renderWindow.Render()

    return


# This function converts a single track to a vtkActor
def trk2vtkActor(trk, replace):
    # convert trk to vtkPolyData
    trk = np.transpose(np.asarray(trk))
    numberOfPoints = trk.shape[0]

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    colors = vtk.vtkFloatArray()
    colors.SetNumberOfComponents(4)
    colors.SetName("tangents")

    k = 0
    lines.InsertNextCell(numberOfPoints)
    for j in range(numberOfPoints):
        points.InsertNextPoint(trk[j, :])
        lines.InsertCellPoint(k)
        k = k + 1

        if j < (numberOfPoints - 1):
            direction = trk[j + 1, :] - trk[j, :]
            direction = direction / np.linalg.norm(direction)
            colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))
        else:
            colors.InsertNextTuple(np.abs([direction[0], direction[1], direction[2], 1]))

    trkData = vtk.vtkPolyData()
    trkData.SetPoints(points)
    trkData.SetLines(lines)
    trkData.GetPointData().SetScalars(colors)

    # make it a tube
    trkTube = vtk.vtkTubeFilter()
    trkTube.SetRadius(0.1)
    trkTube.SetNumberOfSides(4)
    trkTube.SetInputData(trkData)
    trkTube.Update()

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
        transform_filt.SetInputConnection(trkTube.GetOutputPort())
        transform_filt.Update()

    # mapper
    trkMapper = vtk.vtkPolyDataMapper()
    trkMapper.SetInputData(trkTube.GetOutput())

    # actor
    trkActor = vtk.vtkActor()
    trkActor.SetMapper(trkMapper)

    return trkActor


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True

    data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'

    mask_file = b'sub-P0_dwi_mask.nii'
    mask_path = os.path.join(data_dir, mask_file)

    fod_file = b'sub-P0_dwi_FOD.nii'
    fod_path = os.path.join(data_dir, fod_file)

    img_file = b'sub-P0_T1w_biascorrected.nii'
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

    # tracker = Trekker.tracker(trk_path)
    #
    # repos = [0., 0., 0., 0., 0., 0.]
    # brain_actor = load_stl(brain_path, ren, opacity=0.5, replace=repos, user_matrix=np.identity(4))
    #
    # # Add axes to scene origin
    # if SHOW_AXES:
    #     add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
    #     add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
    #     add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])
    #
    # # Show tracks
    # repos_trk = [0., 0., 0., 0., 90., 0.]
    # for i in range(5):
    #     seed = np.array([[-8.49, -8.39, 2.5]])
    #     visualizeTracks(ren, ren_win, tracker, seed, replace=repos_trk, user_matrix=np.linalg.inv(affine))

    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Show tracks
    tracker = Trekker.tracker(fod_path)
    repos_trk = [0., 0., 0., 0., 0., 0.]
    for i in range(5):
        seed = np.array([[-8.49, -8.39, 2.5]])
        visualizeTracks(ren, ren_win, tracker, seed, replace=repos_trk, user_matrix=np.linalg.inv(np.identity(4)))

    # Assign actor to the renderer
    brain_actor = Brain(img_path, mask_path).get_actor(1)

    ren.AddActor(brain_actor)

    # Enable user interface interactor
    iren.Initialize()
    ren_win.Render()
    iren.Start()


if __name__ == '__main__':
    main()
