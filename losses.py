# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:31:23 2019

@author: T_ESTIENNE
"""
import keras
from keras import backend as K
import keras.losses


def zero_loss(y_true, y_pred):
    return K.constant(0., 'float32')


def mean_squared_error_with_zero(y_true, y_pred):
    return keras.losses.mean_squared_error(0., y_pred)


def _dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def averaged_dice_loss(y_true, y_pred):

    zipped = K.concatenate(
        (K.expand_dims(y_true, 0), K.expand_dims(y_pred, 0)), axis=0)
    res = K.mean(K.map_fn(lambda pair: _dice_loss(pair[0], pair[1]),
                          K.permute_dimensions(zipped, (5, 0, 1, 2, 3, 4))), axis=0)
    return res
