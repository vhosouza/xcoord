"""
Basic Visualization
===================
Visualize freesurface surface in a VTK scene
"""

import nibabel.freesurfer.io as fsio
import numpy as np
from stl import mesh

subject_id = 'fsaverage'
hemi = 'lh'
surf = 'inflated'

fil = r'C:\Users\victo\OneDrive\data\nexstim_coord\freesurfer\ppM1_S1\surf\lh.pial'

coord, faces, volume_info = fsio.read_geometry(fil, read_metadata=True)

# Using an existing stl file:
your_mesh = mesh.Mesh.from_file('head.stl')

# Or creating a new mesh (make sure not to overwrite the `mesh` import by
# naming it `mesh`):
VERTICE_COUNT = 100
data = numpy.zeros(VERTICE_COUNT, dtype=mesh.Mesh.dtype)
your_mesh = mesh.Mesh(data, remove_empty_areas=False)

# The mesh normals (calculated automatically)
your_mesh.normals
# The mesh vectors
your_mesh.v0, your_mesh.v1, your_mesh.v2
# Accessing individual points (concatenation of v0, v1 and v2 in triplets)
assert (your_mesh.points[0][0:3] == your_mesh.v0[0]).all()
assert (your_mesh.points[0][3:6] == your_mesh.v1[0]).all()
assert (your_mesh.points[0][6:9] == your_mesh.v2[0]).all()
assert (your_mesh.points[1][0:3] == your_mesh.v0[1]).all()

your_mesh.save('new_stl_file.stl')