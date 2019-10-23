# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:36:05 2019

@author: T_ESTIENNE
"""

import numpy as np
import SimpleITK as sitk
import os
import sys


main_path = os.path.abspath(__file__)
n = main_path.find('Python')
if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)


def recover_from_crop(array, crop_mask):

    output_size = crop_mask.shape

    if len(array.shape) == 4:
        _, depth, height, width = array.shape
    else:
        depth, height, width = array.shape  # Oasis only 3D, no modality

    if depth == output_size[0]:
        depth_min = 0
        depth_max = depth
    else:
        depth_min = int((depth - output_size[0])/2)
        depth_max = -(depth - output_size[0] - depth_min)

    if height == output_size[1]:
        height_min = 0
        height_max = height
    else:
        height_min = int((height - output_size[1])/2)
        height_max = -(height - output_size[1] - height_min)

    if width == output_size[2]:
        width_min = 0
        width_max = width
    else:
        width_min = int((width - output_size[2])/2)
        width_max = -(width - output_size[2] - width_min)


    if len(crop_mask.shape) == 4:
        mask = np.zeros((depth, height, width, crop_mask.shape[-1]))
    else:
        mask = np.zeros((depth, height, width))

    try:
        mask[depth_min:depth_max,
             height_min:height_max,
             width_min:width_max
             ] = crop_mask
    except:
        mask = np.zeros(array.shape[1:]+(crop_mask.shape[-1],))
        mask[depth_min:depth_max, height_min:height_max,
             width_min:width_max, :] = crop_mask

    return mask


def main(patients):
    for patient in patients:

        array = np.load(os.path.join(data_path, patient))

        crop_mask = np.load(os.path.join(pred_path, patient))

        crop_mask = np.argmax(crop_mask, axis=-1)

        crop_mask[crop_mask == 3] = 4

        mask = recover_from_crop(array, crop_mask)

        img = sitk.GetImageFromArray(mask)
        sitk.WriteImage(img, os.path.join(save_path, patient[:-4] + '.nii.gz'))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python file BRATS_numpy_folder predictions_folder save_folder')
    data_path = os.path.abspath(sys.argv[1])
    pred_path = os.path.abspath(sys.argv[2])
    save_path = os.path.abspath(sys.argv[3])

    patients = [f for f in os.listdir(pred_path) if f.endswith('.npy')]
    print('patients', patients)
    main(patients)


def convert_arrays_to_submission(names_images, numpy_images, numpy_predictions, output_folder, logger=None,
                                 is_segmentation_mask=True, filename_suffix_before_format=None):
    log_op = print if logger is None else logger.info
    for name, npy_img, npy_pred in zip(names_images, numpy_images, numpy_predictions):
        if is_segmentation_mask:
            npy_pred = np.argmax(npy_pred, axis=-1)
            if filename_suffix_before_format == 'decoded_source':
                npy_pred[npy_pred == 3] = 4

        res_npy = recover_from_crop(npy_img, npy_pred)
        res = sitk.GetImageFromArray(res_npy)
        filename_no_format = name[:-4]
        if filename_suffix_before_format is not None and filename_suffix_before_format is not '':
            filename_no_format = filename_no_format + '_' + filename_suffix_before_format

        output_filepath = os.path.join(
            output_folder, filename_no_format + '.nii.gz')
        sitk.WriteImage(res, output_filepath)
        log_op('Converted {} into {}; shape {}'.format(
            name, filename_no_format + '.nii.gz', res_npy.shape))
