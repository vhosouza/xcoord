#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np
import external.transformations as tf
import Trekker
import vtk
import time
import psutil
import dti_funcs as dti


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True
    COMPUTE_TRACTS = True
    n_tracts = 240
    # n_tracts = 24
    n_threads = 2*psutil.cpu_count()
    img_shift = 0  # 255

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'

    filenames = {'T1': 'sub-S1_ses-S8741_T1w', 'FOD': 'FOD_T1_space', 'ACT': 'trekkerACTlabels',
                 'COIL': 'magstim_fig8_coil', 'HEAD': 'head_inv', 'BRAIN': 'brain_inv', 'BRAINSIM': 'gm', 'WM': 'skin'}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    trk_path = os.path.join(data_dir, filenames['FOD'] + '.nii')
    act_path = os.path.join(data_dir, filenames['ACT'] + '.nii')
    coil_path = os.path.join(data_dir, filenames['COIL'] + '.stl')
    head_inv_path = os.path.join(data_dir, filenames['HEAD'] + '.stl')
    brain_inv_path = os.path.join(data_dir, filenames['BRAIN'] + '.stl')
    brain_sim_path = os.path.join(data_dir, filenames['BRAINSIM'] + '.stl')
    wm_sim_path = os.path.join(data_dir, filenames['WM'] + '.stl')

    imagedata = nb.squeeze_image(nb.load(img_path))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    act_data = nb.squeeze_image(nb.load(act_path))
    act_data = nb.as_closest_canonical(act_data)
    act_data.update_header()
    act_data_arr = act_data.get_fdata()

    # print(imagedata.header)
    # print("pix_dim: {}, img_shape: {}".format(pix_dim, img_shape))

    print("Pixel size: \n")
    print(pix_dim)
    print("\nImage shape: \n")
    print(img_shape)

    print("\nSform: \n")
    print(imagedata.get_qform(coded=True))
    print("\nQform: \n")
    print(imagedata.get_sform(coded=True))
    print("\nFall-back: \n")
    print(imagedata.header.get_base_affine())

    if AFFINE_IMG:
        affine = imagedata.affine
        if NO_SCALE:
            scale, shear, angs, trans, persp = tf.decompose_matrix(imagedata.affine)
            affine = tf.compose_matrix(scale=None, shear=shear, angles=angs, translate=trans, perspective=persp)
    else:
        affine = np.identity(4)

    print("affine: {0}\n".format(affine))

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.SetUseDepthPeeling(1)
    ren.SetOcclusionRatio(0.1)
    ren.SetMaximumNumberOfPeels(100)

    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)
    ren_win.SetMultiSamples(0)
    ren_win.SetAlphaBitPlanes(1)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    repos = [0., 0., 0., 0., 0., 0.]
    # brain in invesalius space (STL as exported by invesalius)
    _ = load_stl(head_inv_path, ren, opacity=.7, colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.identity(4))

    _ = load_stl(wm_sim_path, ren, opacity=.7, colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.identity(4))

    # simnibs brain in RAS+ space
    # _ = load_stl(brain_sim_path, ren, opacity=1., colour=[1.0, 0., 0.], replace=repos, user_matrix=np.identity(4))

    # brain in RAS+ space
    inv2ras = affine.copy()
    inv2ras[1, 3] += pix_dim[1] * img_shape[1]
    inv2ras[0, 3] -= 12
    # _ = load_stl(brain_inv_path, ren, opacity=.6, colour="SkinColor", replace=repos, user_matrix=inv2ras)

    # brain in voxel space
    inv2voxel = np.identity(4)
    inv2voxel[1, 3] = inv2voxel[1, 3] + pix_dim[1] * img_shape[1]
    # _ = load_stl(brain_inv_path, ren, opacity=.6, colour=[0.482, 0.627, 0.698], replace=repos, user_matrix=inv2voxel)

    # simnibs brain in RAS+ space
    ras2inv = np.linalg.inv(affine.copy())
    ras2inv[1, 3] -= pix_dim[1] * img_shape[1]
    _ = load_stl(wm_sim_path, ren, opacity=.7, colour=[0.482, 0.627, 0.698], replace=repos, user_matrix=ras2inv)

    repos_1 = [0., 0., 0., 0., 0., 180.]
    # _ = load_stl(wm_sim_path, ren, opacity=.7, colour=[1., 0., 0.], replace=repos_1, user_matrix=np.linalg.inv(affine))

    # create fiducial markers
    # rowise the coordinates refer to: right ear, left ear, nasion
    # fids_inv = np.array([[168.300, 126.600, 97.000],
    #                      [9.000, 120.300, 93.700],
    #                      [90.100, 33.500, 150.000]])
    fids_inv = np.array([[167.7, 120.9, 96.0],
                         [8.2, 122.7, 91.0],
                         [89.0, 18.6, 129.0]])
    fids_inv_vtk = np.array([[167.7, 120.9, 96.0],
                         [8.2, 122.7, 91.0],
                         [89.0, 18.6, 129.0]])

    # from the invesalius exported fiducial markers you have to multiply the Y coordinate by -1 to
    # transform to the regular 3D invesalius space where coil location is saved
    fids_inv_vtk[:, 1] *= -1

    # the following code converts from the invesalius 3D space to the MRI scanner coordinate system
    fids_inv_vtk_w = fids_inv_vtk.copy()
    fids_inv_vtk_w = np.hstack((fids_inv_vtk_w, np.ones((3, 1))))
    fids_scan = np.linalg.inv(ras2inv) @ fids_inv_vtk_w.T
    fids_vis = fids_scan.T[:3, :3]
    # --- fiducial markers

    seed = np.array([60.0, 147.0, 204.0])
    seed_inv = np.array([60.0, -147.0, 204.0])
    coil_pos = [43.00, 155.47, 225.22, -21.00, -37.45, 58.41]
    m_coil = coil_transform_pos(coil_pos)

    # show coil
    repos_coil = [0., 0., 0., 0., 0., 90.]
    # _ = load_stl(coil_path, ren, opacity=.6, replace=repos_coil, colour=[1., 1., 1.], user_matrix=m_coil)

    # create coil vectors
    vec_length = 75
    p1 = m_coil[:-1, -1]
    coil_dir = m_coil[:-1, 0]
    coil_face = m_coil[:-1, 1]
    p2_face = p1 + vec_length * coil_face
    p2_dir = p1 + vec_length * coil_dir
    coil_norm = np.cross(coil_dir, coil_face)
    p2_norm = p1 - vec_length * coil_norm

    add_line(ren, p1, p2_dir, color=[1.0, .0, .0])
    add_line(ren, p1, p2_face, color=[.0, 1.0, .0])
    add_line(ren, p1, p2_norm, color=[.0, .0, 1.0])
    # --- coil vectors

    p1_change = p1.copy()
    p1_change[1] = -p1_change[1]

    # offset = 40
    # coil_norm = coil_norm/np.linalg.norm(coil_norm)
    # coord_offset_nav = p1 - offset * coil_norm

    # convert to world coordinate space to use as seed for fiber tracking
    seed_world = np.append(seed, 1)[np.newaxis, :].T
    seed_world = affine @ seed_world
    seed_world = seed_world[:3, 0, np.newaxis].T

    # convert to world coordinate space to use as seed for fiber tracking
    seed_world_true = np.append(seed_inv, 1)[np.newaxis, :].T
    seed_world_true = inv2ras @ seed_world_true
    seed_world_true = seed_world_true[:3, 0, np.newaxis].T

    # convert to voxel coordinate space
    seed_mri = np.append(seed_inv, 1)[np.newaxis, :].T
    seed_mri = inv2voxel @ seed_mri
    seed_mri = seed_mri[:3, 0, np.newaxis].T

    # 0: red, 1: green, 2: blue, 3: maroon (dark red),
    # 4: purple, 5: teal (petrol blue), 6: yellow, 7: orange
    colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
               [0.45, 0., 0.5], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

    # for n in range(3):
    #     _ = add_marker(fids_inv[n, :], ren, colours[n], radius=2)

    for n in range(3):
        _ = add_marker(fids_inv_vtk[n, :], ren, colours[n], radius=2)

    for n in range(3):
        _ = add_marker(fids_vis[n, :], ren, colours[n], radius=2)

    _ = add_marker(p1, ren, colours[4], radius=2)
    _ = add_marker(seed_inv, ren, colours[5], radius=2)
    _ = add_marker(np.squeeze(seed_world), ren, colours[6], radius=2)
    _ = add_marker(np.squeeze(seed_world_true), ren, colours[3], radius=2)
    _ = add_marker(seed, ren, colours[7], radius=2)
    _ = add_marker(np.squeeze(seed_mri), ren, colours[1], radius=2)

    # create tracts
    if COMPUTE_TRACTS:
        # Show tracks
        repos_trk = [0., -(pix_dim[1] * img_shape[1]), 0., 0., 0., 0.]

        matrix_vtk = vtk.vtkMatrix4x4()
        trans = np.identity(4)
        trans[1, -1] = repos_trk[1]
        final_matrix = np.linalg.inv(affine) @ trans
        print("final_matrix: {}".format(final_matrix))

        for row in range(0, 4):
            for col in range(0, 4):
                matrix_vtk.SetElement(row, col, final_matrix[row, col])

        root = vtk.vtkMultiBlockDataSet()

        start_time = time.time()
        tracker = Trekker.initialize(bytes(trk_path, 'utf-8'))
        tracker.seed_maxTrials(1)
        tracker.minFODamp(0.1)
        tracker.writeInterval(50)
        tracker.maxLength(200)
        tracker.minLength(20)
        tracker.maxSamplingPerStep(100)
        tracker.numberOfThreads(n_threads)
        duration = time.time() - start_time
        print("Initialize Trekker: {:.2f} ms".format(1e3 * duration))

        count_tracts = 0
        start_time_all = time.time()

        for n in range(round(n_tracts/n_threads)):
            # branch = dti.multi_block(tracker, seed, n_threads)
            branch = dti.multi_block(tracker, seed_world_true, n_threads)
            count_tracts += branch.GetNumberOfBlocks()

            # start_time = time.time()
            # root = dti.tracts_root(out_list, root, n)
            root.SetBlock(n, branch)
            # duration = time.time() - start_time
            # print("Compute root {}: {:.2f} ms".format(n, 1e3*duration))

        duration = time.time() - start_time_all
        print("Compute multi {}: {:.2f} ms".format(n, 1e3*duration))
        print("Number computed tracts {}".format(count_tracts))
        print("Number computed branches {}".format(root.GetNumberOfBlocks()))

        start_time = time.time()
        tracts_actor = dti.compute_actor(root, matrix_vtk)
        duration = time.time() - start_time
        print("Compute actor: {:.2f} ms".format(1e3*duration))

        ren.AddActor(tracts_actor)

    # Add axes to scene origin
    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Enable user interface interactor
    iren.Initialize()
    ren_win.Render()
    iren.Start()


def load_stl(stl_path, ren, opacity=1., visibility=1, position=False, colour=False, replace=False, user_matrix=np.identity(4)):
    vtk_colors = vtk.vtkNamedColors()
    vtk_colors.SetColor("SkinColor", [233, 200, 188, 255])
    vtk_colors.SetColor("BkgColor", [51, 77, 102, 255])

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    poly_normals = vtk.vtkPolyDataNormals()
    poly_normals.SetInputData(reader.GetOutput())
    poly_normals.ConsistencyOn()
    poly_normals.AutoOrientNormalsOn()
    poly_normals.SplittingOff()
    poly_normals.Update()

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
        transform_filt.SetInputConnection(poly_normals.GetOutputPort())
        transform_filt.Update()

    mapper = vtk.vtkPolyDataMapper()

    if vtk.VTK_MAJOR_VERSION <= 5:
        if replace:
            mapper.SetInput(transform_filt.GetOutput())
        else:
            mapper.SetInput(poly_normals.GetOutput())
    else:
        if replace:
            mapper.SetInputConnection(transform_filt.GetOutputPort())
        else:
            mapper.SetInputConnection(poly_normals.GetOutputPort())

    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.SetVisibility(visibility)
    actor.GetProperty().SetBackfaceCulling(1)

    # outline
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(transform_filt.GetOutputPort())
    mapper_outline = vtk.vtkPolyDataMapper()
    mapper_outline.SetInputConnection(outline.GetOutputPort())
    actor_outline = vtk.vtkActor()
    actor_outline.SetMapper(mapper_outline)

    if colour:
        if type(colour) is str:
            actor.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d(colour))
            actor.GetProperty().SetSpecular(.3)
            actor.GetProperty().SetSpecularPower(20)
            actor_outline.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d("SkinColor"))
            actor_outline.GetProperty().SetSpecular(.3)
            actor_outline.GetProperty().SetSpecularPower(20)

        else:
            actor.GetProperty().SetColor(colour)
            actor_outline.GetProperty().SetColor(colour)

    if position:
        actor.SetPosition(position)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    actor.SetUserMatrix(matrix_vtk)
    actor_outline.SetUserMatrix(matrix_vtk)

    # Assign actor to the renderer
    ren.AddActor(actor)
    ren.AddActor(actor_outline)

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


def add_marker(coord, ren, color, radius):
    # x, y, z = coord

    ball_ref = vtk.vtkSphereSource()
    ball_ref.SetRadius(radius)
    ball_ref.SetCenter(coord)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(ball_ref.GetOutputPort())

    prop = vtk.vtkProperty()
    prop.SetColor(color)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)

    ren.AddActor(actor)

    return actor


def coil_transform_pos(pos):
    pos[1] = -pos[1]
    a, b, g = np.radians(pos[3:])
    r_ref = tf.euler_matrix(a, b, g, 'sxyz')
    t_ref = tf.translation_matrix(pos[:3])
    m_img = tf.concatenate_matrices(t_ref, r_ref)

    return m_img


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main()
