# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:24:34 2019

@author: T_ESTIENNE

Transformations
"""
import numpy as np


def translate(img, translation=None):
    if translation is None:
        if len(img.shape) == 3:
            x, y, z = np.where(img > 0)
        elif len(img.shape) == 4:
            x, y, z, _ = np.where(img > 0)
        elif len(img.shape) == 5:  # multi modalities
            _, x, y, z, _ = np.where(img[0, ...] > 0)
        else:
            raise ValueError('transformations.translate: found image of {} dimensions not supported'.format(len(img.shape)))

        x_center = int(np.mean(x))
        y_center = int(np.mean(y))
        z_center = int(np.mean(z))

        center = [int(k // 2) for k in img.shape]
        translation = (center[0] - x_center, center[1] - y_center, center[2] - z_center)

    padd = np.max([np.abs(t) for t in translation])
    
    if len(img.shape) == 3:
        n, m, p = img.shape
        new_img = np.zeros((n + 2 * padd, m + 2 * padd, p + 2 * padd), img.dtype)
        new_img[padd: padd + n, padd:padd + m, padd:padd + p] = img
    elif len(img.shape) == 4:
        n, m, p, q = img.shape
        new_img = np.zeros((n + 2 * padd, m + 2 * padd, p + 2 * padd, q), img.dtype)
        new_img[padd: padd + n, padd:padd + m, padd:padd + p, :] = img
    elif len(img.shape) == 5:
        modalities, n, m, p, q = img.shape
        new_img = np.zeros((modalities, n + 2 * padd, m + 2 * padd, p + 2 * padd, q), img.dtype)
        new_img[:, padd: padd + n, padd:padd + m, padd:padd + p, :] = img
    else:
        raise ValueError('transformations.translate: found image of {} dimensions not supported'.format(len(img.shape)))
    
    x = padd - translation[0]
    y = padd - translation[1]
    z = padd - translation[2]
    
    if len(img.shape) == 3:
        new_img = new_img[x:x + n, y:y + m, z:z + p]
    elif len(img.shape) == 4:
        new_img = new_img[x:x + n, y:y + m, z:z + p, :]
    elif len(img.shape) == 5:
        new_img = new_img[:, x:x + n, y:y + m, z:z + p, :]

    return new_img, translation


def center_crop(array, output_size, n_modalities=1):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    depth, height, width, _ = array.shape

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

    # 1 modality
    crop = array[depth_min:depth_max, height_min:height_max, width_min:width_max, :]

    return crop


def random_crop(array, output_size):

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    depth, height, width, _ = array.shape

    if depth == output_size[0]:
        i = 0
    else:
        i = np.random.randint(0, depth - output_size[0])

    if height == output_size[1]:
        j = 0
    else:
        j = np.random.randint(0, height - output_size[1])

    if width == output_size[2]:
        k = 0
    else:
        k = np.random.randint(0, width - output_size[2])

    array = array[i:i + output_size[0],
                  j:j + output_size[1],
                  k:k + output_size[2],
                  :]

    return array


def normalize(array):
    array = array.astype(np.float32)
    for modality in range(array.shape[-1]):
        array[..., modality] = (array[..., modality] - np.mean(array[..., modality])) / np.std(array[..., modality])

    return array
