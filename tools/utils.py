# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:48:29 2019

@author: T_ESTIENNE
"""
import os
import numpy as np
import SimpleITK as sitk
import keras.models as models

# My package
from frontiers_brain import blocks
from frontiers_brain.diffeomorphicTransformer import BuildRegistrationMap, diffeomorphicTransformer3D
from frontiers_brain.main import zero_loss, mean_squared_error_with_zero


def load_nifti(path):
    if path.endswith('.nii.gz'):
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif path.endswith('.npy'):
        img = np.load(path)
        if len(img) == 4:
            return np.transpose(img, (1, 2, 3, 0))
        return img
    else:
        raise ValueError


def save_nifti(np_array, dest_filename):
    img = sitk.GetImageFromArray(np_array)
    sitk.WriteImage(img, dest_filename)


def extract_modalities_image(image_filepath):
    print(image_filepath)
    multi_modalities_img = load_nifti(image_filepath)
    assert len(multi_modalities_img.shape) == 4, multi_modalities_img.shape
    if multi_modalities_img.shape[-1] != 4:
        return os.path.dirname(image_filepath)
    print(multi_modalities_img.shape)
    labels = ['flair', 't1', 't1gado', 't2']
    for i, label in enumerate(labels):
        if i != 1:
            continue
        modality = multi_modalities_img[..., i]
        dest_filename = os.path.join(os.path.dirname(image_filepath),
                                     '.'.join(os.path.basename(image_filepath).split('.')[:-1]) + '_{}.nii.gz'.format(
                                         label))
        save_nifti(modality, dest_filename)
    return os.path.dirname(image_filepath)


def extract_modalities_folder(folderpath):
    niftis = list(map(lambda f: os.path.join(folderpath, f),
                      list(filter(lambda s: s.endswith('deformed_source.nii.gz') or s.endswith('.npy'), os.listdir(folderpath)))))
    for nifti in niftis:
        output_dir = extract_modalities_image(nifti)
        print('extracted modalities of {} into {}'.format(nifti, output_dir))


def load_model_v2(model_path):
    from frontiers_brain.main import averaged_dice_loss, averaged_binary_crossentropy, dice_loss_monochannel

    custom_objects = {'averaged_dice_loss': averaged_dice_loss,
                      'averaged_binary_crossentropy': averaged_binary_crossentropy,
                      'loss': dice_loss_monochannel(0),  # this is for all dice measures
                      'BuildRegistrationMap': BuildRegistrationMap,
                      'diffeomorphicTransformer3D': diffeomorphicTransformer3D,
                      'zero_loss': zero_loss,
                      'mean_squared_error_with_zero': mean_squared_error_with_zero, 
                      'DefReg': blocks.DefReg
                      }

    model = models.load_model(model_path, custom_objects=custom_objects)
    return model