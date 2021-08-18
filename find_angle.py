import numpy as np
import vtk


# Read all the data from the file
# reader = vtk.vtkXMLPolyDataReader()
# reader.SetFileName("peel.vtp")
# reader.Update()

reader = vtk.vtkSTLReader()
reader.SetFileName("peel.stl")
reader.Update()
#
# # Visualize
# mapper = vtk.vtkPolyDataMapper()
# mapper.SetInputConnection(reader.GetOutputPort())

# vtkReader = vtk.vtkPolyDataReader()
# vtkReader.SetFileName("peel.vtp")
# vtkReader.Update()

peel = reader.GetOutput() # Peel

# Compute centers of triangles
centerComputer = vtk.vtkCellCenters()     # This computes ceners of the triangles on the peel
centerComputer.SetInputData(peel)
centerComputer.Update()
peel_centers = vtk.vtkFloatArray()        # This stores the centers for easy access
peel_centers = centerComputer.GetOutput()

# Compute normals of triangles
normalComputer = vtk.vtkPolyDataNormals()  # This computes normals of the triangles on the peel
normalComputer.SetInputData(peel)
normalComputer.ComputePointNormalsOff()
normalComputer.ComputeCellNormalsOn()
normalComputer.Update()
peel_normals = vtk.vtkFloatArray()       # This converts to the normals to an array for easy access
peel_normals = normalComputer.GetOutput().GetCellData().GetNormals()

# Initialize locator
locator = vtk.vtkCellLocator()  # This one will later find the triangle on the peel surface where the coil's normal intersect
locator.SetDataSet(peel)
locator.BuildLocator()

#  ↑↑   This part will be done offline for each peel, when the peels are precomputed
#----------------------------------------------------------------------------------




#----------------------------------------------------------------------------------
#  ↓↓   This part will be done online

cc = np.array((-30, -60, 97)) # Coil center
no = np.array((0.1, 0, -0.9))     # Coil normal (i.e., normal at the center)
cd = np.array((0, 1, 0))      # Coil direction
cr = 50                     # Coil range, e.g., 40 mm from the coil center
er = cc + cr*no             # End of coil's reach


lineSource = vtk.vtkLineSource() # We first create a line which
lineSource.SetPoint1(cc)         # starts at the coil center and
lineSource.SetPoint2(er)         # ends at the end of coil's reach
lineSource.Update()

intersectingCellIds = vtk.vtkIdList()  # This find store the triangles that intersect the coil's normal
locator.FindCellsAlongLine(cc, er, .001, intersectingCellIds)


# In our case, there will mostly be a single triangle. But the function above actually finds every triangle on the surface that intersects
# So in some case, there might be many triangles that are found. To address this issue, we should find the closest triangle and use the normal
# of that, which is done below:

closestPoint = np.array((np.Inf,np.Inf,np.Inf))
closestDist = np.Inf
for i in range(intersectingCellIds.GetNumberOfIds()):
    cellId = intersectingCellIds.GetId(i)
    point = np.array(peel_centers.GetPoint(cellId))
    distance = np.linalg.norm(point-cc)
    if distance < closestDist:
        closestDist = distance
        closestPoint = point
        normal = np.array(peel_normals.GetTuple(cellId))
        angle = np.rad2deg(np.arccos(np.dot(normal,-no)))
    
print(angle)



#----------------------------------------------------------------------------------
#  ↓↓   This part just draws things

# Create a disk to show target
disk = vtk.vtkDiskSource()
disk.SetInnerRadius(2)
disk.SetOuterRadius(4)
disk.SetRadialResolution(100)
disk.SetCircumferentialResolution(100)
disk.Update()

disk_mapper = vtk.vtkPolyDataMapper()
disk_mapper.SetInputData(disk.GetOutput())
disk_actor = vtk.vtkActor()
disk_actor.SetMapper(disk_mapper)
disk_actor.GetProperty().SetColor(1,0,0)
disk_actor.GetProperty().SetOpacity(0.4)
disk_actor.SetPosition(closestPoint[0],closestPoint[1],closestPoint[2])
disk_actor.SetOrientation(cd[0],cd[1],cd[2])

# Create an arrow to show coil direction
arrow = vtk.vtkArrowSource()
arrow.Update()

arrow_mapper = vtk.vtkPolyDataMapper()
arrow_mapper.SetInputData(arrow.GetOutput())
arrow_actor = vtk.vtkActor()
arrow_actor.SetMapper(arrow_mapper)
arrow_actor.GetProperty().SetColor(0, 0, 1)
arrow_actor.GetProperty().SetOpacity(0.4)
arrow_actor.SetScale(5)
arrow_actor.SetPosition(closestPoint[0], closestPoint[1], closestPoint[2])
arrow_actor.SetOrientation(normal[0], normal[1], normal[2])

# Show things
wm_mapper = vtk.vtkPolyDataMapper()
wm_mapper.SetInputData(peel)
wm_actor = vtk.vtkActor()
wm_actor.SetMapper(wm_mapper)
wm_actor.GetProperty().SetOpacity(1.)

line_mapper = vtk.vtkPolyDataMapper()
line_mapper.SetInputConnection(lineSource.GetOutputPort())
line_actor = vtk.vtkActor()
line_actor.SetMapper(line_mapper)

ren = vtk.vtkRenderer()
ren.SetBackground(.1, .2, .5)
ren.AddActor(wm_actor)
ren.AddActor(line_actor)
ren.AddActor(disk_actor)
ren.AddActor(arrow_actor)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(480, 480)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()

