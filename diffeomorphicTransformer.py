#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:17:30 2019

@author: theoestienne

This code is a modification of a previous code from Stergios Christodoulidis
Written for the article "Linear and Deformable Image Registration
with 3D Convolutional Neural Networks", Image Analysis for Moving Organ, Breast, and Thoracic Images

"""

import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec


class DefReg(regularizers.Regularizer):
    """Regularizer for the deformation field
    # Arguments
        alpha: Float; regularization factor.
        value: Float; penalize if different than value
    """

    def __init__(self, alpha=1e-5, value=0):
        self.alpha = K.cast_to_floatx(alpha)
        self.value = K.cast_to_floatx(value)

    def __call__(self, x):
        regularization = self.alpha * K.sum(K.abs(x - self.value))
        return regularization

    def get_config(self):
        return {'alpha': float(self.alpha)}
    
class intergral3DGrid(Layer):
    def __init__(self, mode=0, **kwargs):
        self.mode = mode # mode 0 or 1, T_Affine(T_Deformable(img)) or T_Deformable(T_Affine(img))
        super(intergral3DGrid, self).__init__(**kwargs)

    def _integral3DImage(self, x):
        x_s = K.cumsum(x[..., 0]*2, axis=1)
        y_s = K.cumsum(x[..., 1]*2, axis=2)
        z_s = K.cumsum(x[..., 2]*2, axis=3)
        out = K.stack([x_s, y_s, z_s], axis=-1)
        return out

    def call(self, xx, mask=None):
        if len(xx) == 3:
            [im, defgrad, affine] = xx # defgrad in range [-1,1]
        else:
            [im, defgrad] = xx # defgrad in range [-1,1]
            
        # intergrate spatial gradients
        defgrid = self._integral3DImage(defgrad) # defgrid in range [-1,1]

        # constants
        samples = tf.shape(im)[0]
        x_dim = tf.shape(im)[1]
        y_dim = tf.shape(im)[2]
        z_dim = tf.shape(im)[3]
        channels = tf.shape(im)[4]

        try:
            # apply affine transformation
            identity = tf.tile(tf.constant([[1,0,0,0,0,1,0,0,0,0,1,0]], shape=[1,12], dtype='float32'), (samples,1))
            affine = tf.reshape(affine, (-1, 12)) + identity
            affine = tf.reshape(affine, (samples, 3, 4))
            defgrid = tf.concat((defgrid, tf.ones((samples, x_dim, y_dim, z_dim, 1))), -1)
            defgrid = tf.matmul(tf.reshape(defgrid, (samples, -1, 4)), affine, transpose_b=True)
            defgrid = tf.reshape(defgrid, (samples, x_dim, y_dim, z_dim, 3))
        except:
            pass

        return defgrid


class diffeomorphicTransformer3D(Layer):
    def __init__(self, **kwargs):
        super(diffeomorphicTransformer3D, self).__init__(**kwargs)
        
    def _repeat(self, x, n_repeats):
        rep = tf.expand_dims(tf.ones(n_repeats, tf.int32), 0)
        x = tf.tensordot(tf.reshape(x, [-1, 1]), rep, axes=1)
        return tf.reshape(x, [-1]) #flatten
    
    def call(self, xx, mask=None):
        
        [im, defgrid] = xx
        
        # constants
        samples = tf.shape(im)[0]
        x_dim = tf.shape(im)[1]
        y_dim = tf.shape(im)[2]
        z_dim = tf.shape(im)[3]
        channels = tf.shape(im)[4]
        
        # 
        x_s, y_s, z_s = defgrid[..., 0], defgrid[..., 1], defgrid[..., 2]
        
        x = tf.reshape(x_s, [-1]) #flatten
        y = tf.reshape(y_s, [-1]) #flatten
        z = tf.reshape(z_s, [-1]) #flatten
        
        x_dim_f = tf.cast(x_dim, K.floatx())
        y_dim_f = tf.cast(y_dim, K.floatx())
        z_dim_f = tf.cast(z_dim, K.floatx())
        out_x_dim = tf.cast(x_dim_f, 'int32')
        out_y_dim = tf.cast(y_dim_f, 'int32')
        out_z_dim = tf.cast(z_dim_f, 'int32')
        zero = tf.zeros([], dtype='int32')
        max_x = tf.cast(x_dim - 1, 'int32')
        max_y = tf.cast(y_dim - 1, 'int32')
        max_z = tf.cast(z_dim - 1, 'int32')
        
        # do sampling, pixels on a grid
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), 'int32')
        z1 = z0 + 1
        
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)
        
        dim3 = z_dim
        dim2 = z_dim*y_dim
        dim1 = x_dim*y_dim*z_dim
        
        base = self._repeat(tf.range(samples)*dim1, out_x_dim*out_y_dim*out_z_dim)
        idx_a = base + x0*dim2 + y0*dim3 + z0
        idx_b = base + x0*dim2 + y0*dim3 + z1
        idx_c = base + x0*dim2 + y1*dim3 + z0
        idx_d = base + x0*dim2 + y1*dim3 + z1
        idx_e = base + x1*dim2 + y0*dim3 + z0
        idx_f = base + x1*dim2 + y0*dim3 + z1
        idx_g = base + x1*dim2 + y1*dim3 + z0
        idx_h = base + x1*dim2 + y1*dim3 + z1
        
        # use indices to lookup pixels in the flat
        # image and restore channels dim
        im_flat = tf.reshape(im, [-1, channels])
        Ia = tf.gather_nd(im_flat, tf.expand_dims(idx_a, 1)) # 000
        Ib = tf.gather_nd(im_flat, tf.expand_dims(idx_b, 1)) # 001
        Ic = tf.gather_nd(im_flat, tf.expand_dims(idx_c, 1)) # 010
        Id = tf.gather_nd(im_flat, tf.expand_dims(idx_d, 1)) # 011
        Ie = tf.gather_nd(im_flat, tf.expand_dims(idx_e, 1)) # 100
        If = tf.gather_nd(im_flat, tf.expand_dims(idx_f, 1)) # 101
        Ig = tf.gather_nd(im_flat, tf.expand_dims(idx_g, 1)) # 110
        Ih = tf.gather_nd(im_flat, tf.expand_dims(idx_h, 1)) # 111
        
        # and finanly calculate trilinear interpolation
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        x0_f = tf.cast(x0, K.floatx())
        x1_f = tf.cast(x1, K.floatx())
        y0_f = tf.cast(y0, K.floatx())
        y1_f = tf.cast(y1, K.floatx())
        z0_f = tf.cast(z0, K.floatx())
        z1_f = tf.cast(z1, K.floatx())
        
        xd = tf.expand_dims(x-x0_f, 1)
        yd = tf.expand_dims(y-y0_f, 1)
        zd = tf.expand_dims(z-z0_f, 1)
        
        Cae = Ia*(1-xd) + Ie*xd
        Cbf = Ib*(1-xd) + If*xd
        Ccg = Ic*(1-xd) + Ig*xd
        Cdh = Id*(1-xd) + Ih*xd
        
        Caecg = Cae*(1-yd) + Ccg*yd
        Cbfdh = Cbf*(1-yd) + Cdh*yd
        
        output = Caecg*(1-zd) + Cbfdh*zd
        
        output = tf.reshape(output, [samples, x_dim, y_dim, z_dim, channels])
                
        return K.cast(output, K.floatx())


class BuildRegistrationMap(Layer):
    def __init__(self, **kwargs):
        super(BuildRegistrationMap, self).__init__(**kwargs)

    def call(self, seed_registration_map, mask=None):
        x_s = K.cumsum(seed_registration_map[..., 0] * 2, axis=1)
        y_s = K.cumsum(seed_registration_map[..., 1] * 2, axis=2)
        z_s = K.cumsum(seed_registration_map[..., 2] * 2, axis=3)
        registration_map = K.stack([x_s, y_s, z_s], axis=-1)

        return registration_map
