#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nb
import numpy as np
import transformations as tf
import Trekker
import vtk
import time
import psutil
import dti_funcs as dti
# import nexstim2mri as n2m


def main():
    SHOW_AXES = True
    AFFINE_IMG = True
    NO_SCALE = True
    n_tracts = 240
    # n_tracts = 24
    # n_threads = 2*psutil.cpu_count()
    img_shift = 256  # 255

    data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\baran\anat_reg_improve_20200609'
    data_dir = data_dir.encode('utf-8')
    # FOD_path = 'Baran_FOD.nii'
    # trk_path = os.path.join(data_dir, FOD_path)

    # data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'wm_orig_smooth_world.stl'
    brain_path = os.path.join(data_dir, stl_path)

    # data_dir = b'C:\Users\deoliv1\OneDrive\data\dti'
    stl_path = b'wm_2.stl'
    brain_simnibs_path = os.path.join(data_dir, stl_path)

    stl_path = b'wm.stl'
    brain_inv_path = os.path.join(data_dir, stl_path)

    nii_path = b'Baran_FOD.nii'
    trk_path = os.path.join(data_dir, nii_path)

    nii_path = b'Baran_T1_inFODspace.nii'
    img_path = os.path.join(data_dir, nii_path)

    nii_path = b'Baran_trekkerACTlabels_inFODspace.nii'
    act_path = os.path.join(data_dir, nii_path)

    stl_path = b'magstim_fig8_coil.stl'
    coil_path = os.path.join(data_dir, stl_path)

    imagedata = nb.squeeze_image(nb.load(img_path.decode('utf-8')))
    imagedata = nb.as_closest_canonical(imagedata)
    imagedata.update_header()
    pix_dim = imagedata.header.get_zooms()
    img_shape = imagedata.header.get_data_shape()

    act_data = nb.squeeze_image(nb.load(act_path.decode('utf-8')))
    act_data = nb.as_closest_canonical(act_data)
    act_data.update_header()
    act_data_arr = act_data.get_fdata()

    # print(imagedata.header)

    print("pix_dim: {}, img_shape: {}".format(pix_dim, img_shape))

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
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(800, 800)

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)

    minFODamp = np.arange(0.01, 0.11, 0.01)
    dataSupportExponent = np.arange(0.1, 1.1, 0.1)
    # COMBINATION 1
    # tracker = minFODamp(0.01)
    # tracker = dataSupportExponent(0.1)
    # COMBINATION "n"
    # tracker = minFODamp(0.01 * n)
    # tracker = dataSupportExponent(0.1 * n)

    start_time = time.time()
    trekker_cfg = {'seed_max': 1, 'step_size': 0.1, 'min_fod': 0.1, 'probe_quality': 3,
                      'max_interval': 1, 'min_radius_curv': 0.8, 'probe_length': 0.4,
                      'write_interval': 50, 'numb_threads': '', 'max_lenth': 200,
                      'min_lenth': 20, 'max_sampling_step': 100}
    tracker = Trekker.initialize(trk_path)
    tracker, n_threads = dti.set_trekker_parameters(tracker, trekker_cfg)
    duration = time.time() - start_time
    print("Initialize Trekker: {:.2f} ms".format(1e3*duration))

    repos = [0., -img_shift, 0., 0., 0., 0.]
    # brain_actor = load_stl(brain_inv_path, ren, opacity=1., colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.identity(4))

    # the one always been used
    brain_actor = load_stl(brain_simnibs_path, ren, opacity=1., colour=[1.0, 1.0, 1.0], replace=repos, user_matrix=np.linalg.inv(affine))
    # bds = brain_actor.GetBounds()
    # print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    # invesalius surface
    # repos = [0., 0., 0., 0., 0., 0.]
    # brain_actor = load_stl(brain_inv_path, ren, opacity=.5, colour=[1.0, .5, .5], replace=repos, user_matrix=np.identity(4))

    # repos = [0., 0., 0., 0., 0., 0.]
    # brain_actor_mri = load_stl(brain_path, ren, opacity=.1, colour=[0.0, 1.0, 0.0], replace=repos, user_matrix=np.linalg.inv(affine))
    # bds = brain_actor_mri.GetBounds()
    # print("Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    # repos = [0., 256., 0., 0., 0., 0.]
    # brain_inv_actor = load_stl(brain_inv_path, ren, colour="SkinColor", opacity=0.5, replace=repos, user_matrix=np.linalg.inv(affine))
    # brain_inv_actor = load_stl(brain_inv_path, ren, colour="SkinColor", opacity=.6, replace=repos)
    # bds = brain_inv_actor.GetBounds()
    # print("Reposed: Y length: {} --- Bounds: {}".format(bds[3] - bds[2], bds))

    # Add axes to scene origin
    if SHOW_AXES:
        add_line(ren, [0, 0, 0], [150, 0, 0], color=[1.0, 0.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 150, 0], color=[0.0, 1.0, 0.0])
        add_line(ren, [0, 0, 0], [0, 0, 150], color=[0.0, 0.0, 1.0])

    # Show tracks
    repos_trk = [0., -img_shift, 0., 0., 0., 0.]
    # repos_trk = [0., 0., 0., 0., 0., 0.]

    matrix_vtk = vtk.vtkMatrix4x4()

    trans = np.identity(4)
    trans[1, -1] = repos_trk[1]
    final_matrix = np.linalg.inv(affine) @ trans

    print("final_matrix: {}".format(final_matrix))

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, final_matrix[row, col])

    root = vtk.vtkMultiBlockDataSet()
    # for i in range(10):
    # seed = np.array([[-8.49, -8.39, 2.5]])
    # seed = np.array([[27.53, -77.37, 46.42]])
    # from the invesalius exported fiducial markers you have to multiply the Y coordinate by -1 to
    # transform to the regular 3D invesalius space where coil location is saved
    fids_inv = np.array([[168.300, -126.600, 97.000],
                         [9.000, -120.300, 93.700],
                         [90.100, -33.500, 150.000]])

    for n in range(3):
        fids_actor = add_marker(fids_inv[n, :], ren, [1., 0., 0.], radius=2)

    seed = np.array([[-25.66, -30.07, 54.91]])
    coil_pos = [40.17, 152.28, 235.78, -18.22, -25.27, 64.99]
    m_coil = coil_transform_pos(coil_pos)

    repos = [0., 0., 0., 0., 0., 90.]
    coil_actor = load_stl(coil_path, ren, opacity=.6, replace=repos, colour=[1., 1., 1.], user_matrix=m_coil)
    # coil_actor = load_stl(coil_path, ren, opacity=.6, replace=repos, colour=[1., 1., 1.])

    # create coil vectors
    vec_length = 75
    print(m_coil.shape)
    p1 = m_coil[:-1, -1]
    print(p1)
    coil_dir = m_coil[:-1, 0]
    coil_face = m_coil[:-1, 1]
    p2_face = p1 + vec_length * coil_face

    p2_dir = p1 + vec_length * coil_dir

    coil_norm = np.cross(coil_dir, coil_face)
    p2_norm = p1 - vec_length * coil_norm

    add_line(ren, p1, p2_dir, color=[1.0, .0, .0])
    add_line(ren, p1, p2_face, color=[.0, 1.0, .0])
    add_line(ren, p1, p2_norm, color=[.0, .0, 1.0])

    colours = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., .0, 1.],
               [.5, .5, 0.], [0., .5, .5], [1., 1., 0.], [1., .4, .0]]

    marker_actor = add_marker(p1, ren, colours[0], radius=1)

    # p1_change = n2m.coord_change(p1)
    p1_change = p1.copy()
    p1_change[1] = -p1_change[1]
    # p1_change[1] += img_shift

    marker_actor2 = add_marker(p1_change, ren, colours[1], radius=1)

    offset = 40
    coil_norm = coil_norm/np.linalg.norm(coil_norm)
    coord_offset_nav = p1 - offset * coil_norm

    marker_actor_seed_nav = add_marker(coord_offset_nav, ren, colours[3], radius=1)

    coord_offset_mri = coord_offset_nav.copy()
    coord_offset_mri[1] += img_shift
    marker_actor_seed_nav = add_marker(coord_offset_mri, ren, colours[3], radius=1)

    coord_mri_label = [int(s) for s in coord_offset_mri]
    print("offset MRI: {}, and label: {}".format(coord_mri_label,
                                                 act_data_arr[tuple(coord_mri_label)]))

    offset_list = 10 + np.arange(0, 31, 3)
    coord_offset_list = p1 - np.outer(offset_list, coil_norm)
    coord_offset_list += [0, img_shift, 0]
    coord_offset_list = coord_offset_list.astype(int).tolist()

    # for pt in coord_offset_list:
    #     print(pt)
    #     if act_data_arr[tuple(pt)] == 2:
    #         cl = colours[5]
    #     else:
    #         cl = colours[4]
    #     _ = add_marker(pt, ren, cl)

    x = np.arange(-4, 5, 2)
    y = np.arange(-4, 5, 2)
    z = 10 + np.arange(0, 31, 3)
    xv, yv, zv = np.meshgrid(x, y, - z)
    coord_grid = np.array([xv, yv, zv])

    start_time = time.time()
    for p in range(coord_grid.shape[1]):
        for n in range(coord_grid.shape[2]):
            for m in range(coord_grid.shape[3]):
                pt = coord_grid[:, p, n, m]
                pt = np.append(pt, 1)
                pt_tr = m_coil @ pt[:, np.newaxis]
                pt_tr = np.squeeze(pt_tr[:3]).astype(int) + [0, img_shift, 0]
                pt_tr = tuple(pt_tr.tolist())
                if act_data_arr[pt_tr] == 2:
                    cl = colours[6]
                elif act_data_arr[pt_tr] == 1:
                    cl = colours[7]
                else:
                    cl = [1., 1., 1.]
                # print(act_data_arr[pt_tr])
                _ = add_marker(pt_tr, ren, cl, radius=1)

    duration = time.time() - start_time
    print("Compute coil grid: {:.2f} ms".format(1e3*duration))

    start_time = time.time()
    # create grid of points
    grid_number = x.shape[0]*y.shape[0]*z.shape[0]
    coord_grid = coord_grid.reshape([3, grid_number]).T
    # sort grid from distance to the origin/coil center
    coord_list = coord_grid[np.argsort(np.linalg.norm(coord_grid, axis=1)), :]
    # make the coordinates homogeneous
    coord_list_w = np.append(coord_list.T, np.ones([1, grid_number]), axis=0)
    # apply the coil transformation matrix
    coord_list_w_tr = m_coil @ coord_list_w
    # convert to int so coordinates can be used as indices in the MRI image space
    coord_list_w_tr = coord_list_w_tr[:3, :].T.astype(int) + np.array([[0, img_shift, 0]])
    # extract the first occurrence of a specific label from the MRI image
    labs = act_data_arr[coord_list_w_tr[..., 0], coord_list_w_tr[..., 1], coord_list_w_tr[..., 2]]
    lab_first = np.argmax(labs == 1)
    if labs[lab_first] == 1:
        pt_found = coord_list_w_tr[lab_first, :]
        _ = add_marker(pt_found, ren, [0., 0., 1.], radius=1)
    # convert coordinate back to invesalius 3D space
    pt_found_inv = pt_found - np.array([0., img_shift, 0.])
    # convert to world coordinate space to use as seed for fiber tracking
    pt_found_tr = np.append(pt_found, 1)[np.newaxis, :].T
    pt_found_tr = affine @ pt_found_tr
    pt_found_tr = pt_found_tr[:3, 0, np.newaxis].T
    duration = time.time() - start_time
    print("Compute coil grid fast: {:.2f} ms".format(1e3*duration))

    # create tracts
    count_tracts = 0
    start_time_all = time.time()

    # uncertain_params = list(zip(dataSupportExponent, minFODamp))
    for n in range(0, round(n_tracts/n_threads)):
        # branch = dti.multi_block(tracker, seed, n_threads)
        # branch = dti.multi_block(tracker, pt_found_tr, n_threads)
        # rescale n so that there is no 0 opacity tracts
        n_param = (n % 10) + 1
        branch = dti.multi_block_uncertainty(tracker, pt_found_tr, n_threads, n_param)
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

    # Assign actor to the renderer
    # ren.AddActor(brain_actor)
    # ren.AddActor(brain_inv_actor)
    # ren.AddActor(coil_actor)

    start_time = time.time()
    ren.AddActor(tracts_actor)
    duration = time.time() - start_time
    print("Add actor: {:.2f} ms".format(1e3*duration))
    # ren.AddActor(brain_actor_mri)

    planex, planey, planez = raw_image(act_path, ren)

    planex.SetInteractor(iren)
    planex.On()
    planey.SetInteractor(iren)
    planey.On()
    planez.SetInteractor(iren)
    planez.On()

    _ = add_marker(np.squeeze(seed).tolist(), ren, [0., 1., 0.], radius=1)
    _ = add_marker(np.squeeze(pt_found_tr).tolist(), ren, [1., 0., 0.], radius=1)
    _ = add_marker(pt_found_inv, ren, [1., 1., 0.], radius=1)

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

    if colour:
        if type(colour) is str:
            actor.GetProperty().SetDiffuseColor(vtk_colors.GetColor3d(colour))
            actor.GetProperty().SetSpecular(.3)
            actor.GetProperty().SetSpecularPower(20)

        else:
            actor.GetProperty().SetColor(colour)

    if position:
        actor.SetPosition(position)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, user_matrix[row, col])

    actor.SetUserMatrix(matrix_vtk)

    # Assign actor to the renderer
    ren.AddActor(actor)

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


def raw_image(filepath, renderer):
    mask_reader = vtk.vtkNIFTIImageReader()
    mask_reader.SetFileName(filepath)
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

    #Add outlineactor
    renderer.AddActor(outlineActor)

    #Load widget interactors and enable
    # planeWidgetX.SetInteractor(interactor)
    # planeWidgetX.On()
    # planeWidgetY.SetInteractor(interactor)
    # planeWidgetY.On()
    # planeWidgetZ.SetInteractor(interactor)
    # planeWidgetZ.On()

    return planeWidgetX, planeWidgetY, planeWidgetZ


if __name__ == '__main__':
    main()

# 240 tracts in a single block/run
# Seed coordinates: 0.00 ms
# Run Trekker: 1154.00 ms
# Tracts to array: 2.00 ms
# Tracts directions: 12.00 ms
# Compute tubes: 59.00 ms
# Compute root: 1.00 ms
# Tracts computation: 80.00 ms
# Compute actor: 0.00 ms
# Add actor: 0.00 ms
