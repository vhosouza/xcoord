#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import vtk

import image_funcs as imf


def main():

    # data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\joonas'
    #
    # filenames = {'T1': 'sub-S1_ses-S8741_T1w',
    #              'STL_INVESALIUS': 'half_head_inv',
    #              'STL_WORLD': 'half_head_world'}

    subj_id = 7
    item = 'brain'
    data_dir = os.environ.get('OneDrive') + r'\data\nexstim_coord\mri\ppM1_S{}'.format(subj_id)

    filenames = {'T1': 'ppM1_S{}'.format(subj_id),
                 'STL_INVESALIUS': 'ppM1_S{}_{}_shell_inv'.format(subj_id, item),
                 'STL_WORLD': 'ppM1_S{}_{}_shell_world'.format(subj_id, item)}

    img_path = os.path.join(data_dir, filenames['T1'] + '.nii')
    stl_invesalius_path = os.path.join(data_dir, filenames['STL_INVESALIUS'] + '.stl')
    stl_world_path = os.path.join(data_dir, filenames['STL_WORLD'] + '.stl')

    imagedata, affine = imf.load_image(img_path)
    # mri2inv_mat = imf.mri2inv(imagedata, affine)
    inv2mri_mat = imf.inv2mri(imagedata, affine)

    matrix_vtk = vtk.vtkMatrix4x4()

    for row in range(0, 4):
        for col in range(0, 4):
            matrix_vtk.SetElement(row, col, inv2mri_mat[row, col])

    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_invesalius_path)
    reader.Update()

    # poly_normals = vtk.vtkPolyDataNormals()
    # poly_normals.SetInputData(reader.GetOutput())
    # poly_normals.ConsistencyOn()
    # poly_normals.AutoOrientNormalsOn()
    # poly_normals.SplittingOff()
    # poly_normals.UpdateInformation()
    # poly_normals.Update()

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix_vtk)

    transform_polydata = vtk.vtkTransformPolyDataFilter()
    transform_polydata.SetTransform(transform)
    transform_polydata.SetInputData(reader.GetOutput())
    transform_polydata.Update()

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileTypeToBinary()
    stl_writer.SetFileName(stl_world_path)
    stl_writer.SetInputConnection(transform_polydata.GetOutputPort())
    stl_writer.Write()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.set_printoptions(suppress=True, precision=2)
    main()
