# -----------------------------------------------------------------------------
#
# Load msh file and export each entity as an STL file using Gmsh Python API
#
# Recommended by: http://www.geuz.org/pipermail/gmsh/2018/012525.html
# -----------------------------------------------------------------------------
import os
import gmsh


name_list = {1001: 'wm', 1002: 'gm', 1003: 'csf', 1005: 'skin', 1006: 'eyes',
             1007: 'skull', 1008: 'skull_broken', 1009: 'vessels', 1010: 'ventricles'}

# subject_id = 'EEGTMSseg_082'
subject_id = 'EEGTMSseg_260_T1w_MPR_20230928145653_7'
root_path = r'C:\Users\vhosouza\Downloads\mri\{}'.format(subject_id)
simnibs_dir = 'm2m_{}'.format(subject_id)
export_path = os.path.join(root_path, 'surfaces_stl')

if not os.path.exists(export_path):
    os.makedirs(export_path)

# %%
gmsh.initialize()

gmsh.open(os.path.join(root_path, simnibs_dir, '{}.msh'.format(subject_id)))

# Print the model name and dimension:
print('Model ' + gmsh.model.getCurrent() + ' (' + str(gmsh.model.getDimension()) + 'D)')

# %%
# Geometrical data is made of elementary model `entities', called `points'
# (entities of dimension 0), `curves' (entities of dimension 1), `surfaces'
# (entities of dimension 2) and `volumes' (entities of dimension 3). As we have
# seen in the other Python tutorials, elementary model entities are identified
# by their dimension and by a `tag': a strictly positive identification
# number. Model entities can be either CAD entities (from the built-in `geo'
# kernel or from the OpenCASCADE `occ' kernel) or `discrete' entities (defined
# by a mesh). `Physical groups' are collections of model entities and are also
# identified by their dimension and by a tag.

# Get all the elementary entities in the model, as a vector of (dimension, tag)
# pairs:
entities = gmsh.model.getEntities()

for e in entities:
    # Dimension and tag of the entity:
    dim, tag = e

    if dim == 2:
        if tag in [1002, 1005]:

            gmsh.model.removePhysicalGroups()
            gmsh.model.addPhysicalGroup(dim, [tag], 100, name_list[tag])

            gmsh.write(os.path.join(export_path, "{}.stl".format(name_list[tag])))

# We can use this to clear all the model data:
gmsh.clear()

gmsh.finalize()
