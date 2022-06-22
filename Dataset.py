#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:27:15 2019

@author: theoestienne
"""
import numpy as np
import keras
import os
import SimpleITK as sitk
import nibabel.freesurfer.mghformat as mgh
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
from keras.utils import to_categorical

from frontiers_code import transformations

def aseg_label(all_label=False):
    if all_label:
        return [0, 16, 10, 49, 47, 8, 2, 41, 7, 46, 12, 51, 28, 60, 13, 52, 11,
                50, 4, 43, 17, 53, 14, 15, 18, 54, 3, 42, 24]
    else:

        return [0, 2, 41, 3, 42, 4, 43]


def label_encoder(aseg_label):
    le = preprocessing.LabelEncoder()

    le.fit(aseg_label)

    return le


def brats2oasis(array):
    '''
        From BRATS space to OASIS space
    '''
    tab = np.rollaxis(array, 2, 0)
    tab = tab[::-1, ::-1, ::-1]

    return tab


def oasis2brats(array):
    '''
        From oasis space to BRATS space
    '''
    tab = np.rollaxis(array, 2, 1)
    tab = np.swapaxes(tab, 2, 0)
    tab = tab[::-1, ::-1, ::-1]

    return tab


def load_nifti(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def load_mgh(path):
    return mgh.load(path).get_data()


def load_datasets(path):
    n = len(path)

    nifti = [os.path.join(root, name)[n:]
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith('.npy')
             or name.endswith('.nii.gz')
             or name.endswith('orig.mgz')]

    return nifti


def load_existing_dataset(path, cohort):
    files_train = np.loadtxt(path + cohort + '_train.txt', dtype=str)
    files_val = np.loadtxt(path + cohort + '_val.txt', dtype=str)
    files_test = np.loadtxt(path + cohort + '_test.txt', dtype=str)

    return files_train, files_val, files_test


def create_dataset(files, path, cohort):
    (files_train,
     files_validation) = model_selection.train_test_split(files,
                                                          test_size=0.3,
                                                          random_state=42)

    (files_test,
     files_validation) = model_selection.train_test_split(files_validation,
                                                          test_size=0.3,
                                                          random_state=42)

    np.savetxt(path + cohort + '_train.txt', files_train, fmt='%s')
    np.savetxt(path + cohort + '_val.txt', files_validation, fmt='%s')
    np.savetxt(path + cohort + '_test.txt', files_test, fmt='%s')

    return files_train, files_validation, files_test


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, brats_IDs, data_path, batch_size=4, dim=(64, 64, 64),
                 n_input_channels=1, n_output_channels=1, shuffle=True, validation=False,
                 translation=False, mono_patient=False, to_categorical=False, n_modalities=4,
                 with_source_masks=True, with_target_masks=True, inference=False, cohort='brats',
                 mono_gpu=False, nb_train_files=0, concatenate_inputs=False):
        """Initialization

        :param mono_patient: True to return only image + mask of 1 patient, False to return for a pair of patients"""
        self.batch_size = batch_size
        self.brats_IDs = brats_IDs
        self.data_path = data_path

        if nb_train_files > 0 and not validation:
            self.brats_IDs = brats_IDs[:nb_train_files]

        self.brats_path = data_path + 'BRATS/numpy/'
        self.mask_path = data_path + 'BRATS/mask/'
        self.oasis_path = data_path

        self.inference = inference
        if self.inference:
            self.brats_path = data_path

        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.n_modalities = n_modalities
        self.monochannel_dim = dim
        self.image_dim = (*self.monochannel_dim, n_input_channels)
        self.mask_dim = (*self.monochannel_dim, n_output_channels)

        self.validation = validation
        self.shuffle = shuffle
        self.on_epoch_end()

        self.mono_patient = mono_patient
        self.to_categorical = to_categorical

        self.with_source_masks = with_source_masks
        self.with_target_masks = with_target_masks

        if self.validation:
            self.validation_index()

        ones = np.ones((batch_size, *dim))

        self.identity_grid = np.stack([np.cumsum(ones, axis=1),
                                       np.cumsum(ones, axis=2),
                                       np.cumsum(ones, axis=3)],
                                      axis=-1)

        self.translation = translation

        self.aseg_label = aseg_label(True)
        self.label_encoder = label_encoder(self.aseg_label)

        self.cohort = cohort
        self.mono_gpu = mono_gpu

        self.concatenate_inputs = concatenate_inputs  # True to concatenate channelwise the input source+target, else yield separate images


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.brats_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_moving = self.indexes_moving[index * self.batch_size:(index + 1) * self.batch_size]
        indexes_reference = self.indexes_reference[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [(self.brats_IDs[i], self.brats_IDs[j]) for i, j in zip(indexes_moving, indexes_reference)]

        # Generate data
        if self.mono_patient:
            # moving, moving_mask = self.__data_generation(list_IDs_temp)
            return self.__data_generation(list_IDs_temp)
            # return moving, moving_mask

        input_data, label_data = self.__data_generation(list_IDs_temp)
        batch_source, batch_target = input_data
        if self.inference and self.cohort == 'brats':
            batch_source_mask, batch_target_mask = None, None
        else:
            batch_source_mask, batch_target_mask = label_data

        decoders_keys = ['decoder_segmentation_1', 'decoder_segmentation_2']

        if self.concatenate_inputs:
            concatenated_source_target = np.concatenate((batch_source, batch_target), axis=-1)
            assert concatenated_source_target.shape[:-1] == batch_source.shape[:-1] == batch_target.shape[:-1]
            assert concatenated_source_target.shape[-1] == 2 * batch_source.shape[-1] == 2 * batch_target.shape[-1]
            inputs = [concatenated_source_target, batch_source]
        else:
            inputs = [batch_source, batch_target]

        return inputs, {decoders_keys[0]: batch_source_mask,
                        decoders_keys[1]: batch_target_mask,
                        'registration_map_construction': np.zeros(self.batch_size),
                        'registration_map_application': batch_target,
                        'registration_map_application_losstrick': np.zeros(self.batch_size),
                        'predicted_background_merged_mask': np.zeros(self.batch_size)}

    def validation_index(self):
        self.indexes_moving = list(range(len(self.brats_IDs)))
        self.indexes_reference = list(range(1, len(self.brats_IDs))) + [0]

    def on_epoch_end(self):
        'Updates indexes after each epoch for training'

        if not self.validation:
            self.indexes_moving = np.arange(len(self.brats_IDs))
            self.indexes_reference = np.arange(len(self.brats_IDs))

            if self.shuffle:
                np.random.shuffle(self.indexes_moving)
                np.random.shuffle(self.indexes_reference)

    def _get_image_patient(self, id_image):
        
        if self.cohort == 'brats':
            array_type = 'brats'
            mask_type = 'mask'
            array_path = self.brats_path + id_image
            mask_path = self.mask_path + id_image
        elif self.cohort == 'oasis':
            array_type = 'oasis'
            mask_type = 'aseg'
            array_path = self.oasis_path + id_image
            mask_path = self.oasis_path + id_image

        # Store mri
        array = self.load(array_path, array_type)
        image, translation = self.transform(array)

        # Store mask
        if not self.inference or self.cohort == 'oasis':
            array = self.load(mask_path, mask_type)
            mask, _ = self.transform(array, normalize=False, translation=translation, is_mask=True)
            if self.to_categorical:
                mask = to_categorical(mask, num_classes=self.n_output_channels)
        else:
            mask = None

        if self.n_input_channels == 1 and self.cohort == 'brats':
            image = image[..., [1]]
        return image, mask, translation

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        """ Generates data containing batch_size samples """
        # Initialization
        batch_moving = np.empty((self.batch_size, *self.image_dim))
        batch_moving_mask = np.empty((self.batch_size, *self.mask_dim))
        if not self.mono_patient:
            batch_reference = np.empty((self.batch_size, *self.image_dim))
            batch_reference_mask = np.empty((self.batch_size, *self.mask_dim))
        else:
            batch_reference = None
            batch_reference_mask = None

        # Generate data
        for i, (ID_moving, ID_reference) in enumerate(list_IDs_temp):
            image, mask, translation = self._get_image_patient(ID_moving)
            batch_moving[i, :, :, :, :] = image
            batch_moving_mask[i, :, :, :, :] = mask

            if not self.mono_patient:
                image, mask, _ = self._get_image_patient(ID_reference)
                batch_reference[i, :, :, :, :] = image
                batch_reference_mask[i, :, :, :, :] = mask

        if self.mono_patient:
            if self.inference:
                return batch_moving
            return batch_moving, batch_moving_mask

        return (batch_moving, batch_reference), (batch_moving_mask, batch_reference_mask)

    def transform(self, array, normalize=True, translation=None, is_mask=False):
        if self.translation:
            array, translation = transformations.translate(array, translation)
        else:
            translation = None

        if len(array.shape) == 3:
            array = array[:, :, :, np.newaxis]

        array = transformations.center_crop(array, self.monochannel_dim, n_modalities=self.n_modalities)

        if normalize and not is_mask:
            array = transformations.normalize(array)

        return array, translation

    def load(self, path, type):
        
        if type == 'brats':
            img = np.load(path)[:, :, :, :]
            img = np.transpose(img, (1, 2, 3, 0)).astype(np.float32)  # put modalities channels last
            return img
        elif type == 'mask':
            crude_mask = np.load(path)
            # Merge class 0 (background) and 3 (brain)
            crude_mask[crude_mask == 3] = 0
            # Offset class 4
            crude_mask[crude_mask == 4] = 3
            assert 0 <= np.min(crude_mask) <= np.max(crude_mask) <= 3
            return crude_mask.astype(np.float32)

        elif type == 'oasis':
            brain = load_mgh(path)
            brain = oasis2brats(brain)
            return brain
        elif type == 'aseg':
            aseg = load_mgh(path[:-9] + '_aseg.mgz')
            aseg[~np.isin(aseg, self.aseg_label)] = 0
            aseg = aseg.flatten()
            aseg = self.label_encoder.transform(aseg)
            n_class = len(self.label_encoder.classes_)
            aseg = np.reshape(aseg, (256, 256, 256))
            one_hot = np.eye(n_class)[aseg]
            one_hot = oasis2brats(one_hot)
            return one_hot

        else:
            print('Wrong files')
            return None
