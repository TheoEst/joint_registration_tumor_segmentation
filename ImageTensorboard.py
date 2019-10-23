#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:50:17 2019

@author: theoestienne

https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1
"""

import tensorflow as tf
from PIL import Image
import io
import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


def plot_result(moving, reference, deformed, grid, batch):
    kwargs = {'cmap': 'gray'}

    fig, ax = plt.subplots(3, 4, gridspec_kw={'wspace': 0, 'hspace': 0.02,
                                              'top': 0.93, 'bottom': 0.01,
                                              'left': 0.01, 'right': 0.99})

    x_slice = int(moving.shape[1] // 2)
    y_slice = int(moving.shape[2] // 2)
    z_slice = int(moving.shape[3] // 2)

    ax[0, 0].imshow(reference[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 0].imshow(reference[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 0].imshow(reference[batch, :, :, z_slice, 0], **kwargs)

    ax[0, 1].imshow(moving[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 1].imshow(moving[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 1].imshow(moving[batch, :, :, z_slice, 0], **kwargs)

    ax[0, 2].imshow(deformed[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 2].imshow(deformed[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 2].imshow(deformed[batch, :, :, z_slice, 0], **kwargs)

    dx, dy, dz = (grid[batch, :, :, :, 0],
                  grid[batch, :, :, :, 1],
                  grid[batch, :, :, :, 2])

    ax[0, 3].contour(dy[x_slice, ::-1, :], 100, alpha=0.90, linewidths=0.5)
    ax[0, 3].contour(dz[x_slice, ::-1, :], 100, alpha=0.90, linewidths=0.5)

    ax[1, 3].contour(dx[:, y_slice, :], 100, alpha=0.90, linewidths=0.5)
    ax[1, 3].contour(dz[:, y_slice, :], 100, alpha=0.90, linewidths=0.5)

    ax[2, 3].contour(dx[:, :, z_slice], 100, alpha=0.90, linewidths=0.5)
    ax[2, 3].contour(dy[:, :, z_slice], 100, alpha=0.90, linewidths=0.5)

    for i in range(3):
        for j in range(4):
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    ax[0, 0].set_title('Target')
    ax[0, 1].set_title('Source')
    ax[0, 2].set_title('Deformed')
    ax[0, 3].set_title('Grid')

    fig.canvas.draw()

    plt.close()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def plot_results_for_segmentation(input_image, prediction, ground_truth, invert_masks=True, is_registration=False,
                                  nontumor_mask=None, pred_segmentation_source=None, pred_segmentation_target=None,
                                  gt_segmentation_source=None, gt_segmentation_target=None):
    kwargs = {'cmap': 'gray'}

    # swap black and white for masks
    if invert_masks:
        if is_registration:
            pred_segmentation_source = 1. - pred_segmentation_source
            pred_segmentation_target = 1. - pred_segmentation_target
            gt_segmentation_source = 1. - gt_segmentation_source
            gt_segmentation_target = 1. - gt_segmentation_target
        else:
            prediction = 1. - prediction
            ground_truth = 1. - ground_truth

    x_slice = int(input_image.shape[0] // 2)
    y_slice = int(input_image.shape[1] // 2)
    z_slice = int(input_image.shape[2] // 2)

    def convert_semgentation_pred_to_tumor_mask(pred_segmentation_map):
        return (np.argmax(pred_segmentation_map, axis=-1) == 0).astype(np.float32)[..., np.newaxis]

    fig, ax = plt.subplots(4, 3 if not is_registration or nontumor_mask is None else 8)
    columns_names = ['input', 'prediction', 'ground_truth'] if not is_registration \
        else ['source', 'deformed', 'target', 'pred_so', 'gt_so', 'pred_ta', 'gt_ta', 'pred_sum']
    for r, row in enumerate(ax):
        for c, (column, column_name) in enumerate(zip(row, columns_names)):
            if c == 7 and r > 0:
                pass
            img = input_image[..., r] if c == 0 else prediction[..., r] if c == 1 else ground_truth[..., r] if c == 2 \
                else convert_semgentation_pred_to_tumor_mask(pred_segmentation_source)[..., 0] if c == 3 \
                else gt_segmentation_source[..., r] if c == 4 \
                else convert_semgentation_pred_to_tumor_mask(pred_segmentation_target)[..., 0] if c == 5 \
                else gt_segmentation_target[..., r] if c == 6 \
                else nontumor_mask[..., 0]
            img = img[x_slice, ...]  # select only 1 middle slice
            if c == 0:
                vmin = np.min(img)
                vmax = np.max(img)
                
            if c == 0 or (is_registration and c == 2) or not invert_masks:
                column.imshow(img, **kwargs)

            elif is_registration and c == 1:
                column.imshow(img, cmap=cm.get_cmap('reds'), vmin=0., vmax=3., **kwargs)
            else:
                column.imshow(img, vmin=0., vmax=1., **kwargs)
            if r == 0:
                column.set_title(column_name)
            column.grid(False)
            column.axis('off')
            column.set_xticks([])
            column.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    plt.close()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def plot_results_for_segmentation_registration(input_image, prediction, ground_truth, invert_masks=True,
                                               nontumor_mask=None, pred_segmentation_source=None,
                                               pred_segmentation_target=None,
                                               gt_segmentation_source=None, gt_segmentation_target=None,
                                               deformed_source=None, registration_map=None):
    kwargs = {'cmap': 'gray'}

    # swap black and white for masks
    if invert_masks:
        pred_segmentation_source = 1. - pred_segmentation_source
        pred_segmentation_target = 1. - pred_segmentation_target
        gt_segmentation_source = 1. - gt_segmentation_source
        gt_segmentation_target = 1. - gt_segmentation_target
        if nontumor_mask is not None:
            nontumor_mask = 1. - nontumor_mask

    x_slice = int(input_image.shape[0] // 2)
    y_slice = int(input_image.shape[1] // 2)
    z_slice = int(input_image.shape[2] // 2)

    def convert_semgentation_pred_to_tumor_mask(pred_segmentation_map):
        return (np.argmax(pred_segmentation_map, axis=-1) == 0).astype(np.float32)[..., np.newaxis]

    fig, ax = plt.subplots(5, 8)
    columns_names = ['pS', 'gtS', 'Source', '(R(S)-T)^2', 'R(S)', 'Target', 'pT', 'gtT', 'R']
    for r, row in enumerate(ax):
        for c, (column, column_name) in enumerate(zip(row, columns_names)):
            if r == 4:
                if c not in [2, 3, 4]:
                    column.grid(False)
                    column.axis('off')
                    column.set_xticks([])
                    column.set_yticks([])
                    continue
                else:
                    # Reformat registration grid
                    dx, dy, dz = (registration_map[..., 0], registration_map[..., 1], registration_map[..., 2])
                    if c == 2:
                        column.contour(dy[x_slice, :, :], 100, alpha=0.90, linewidths=0.1)
                        column.contour(dz[x_slice, :, :], 100, alpha=0.90, linewidths=0.1)
                    elif c == 3:
                        column.contour(dx[:, y_slice, :], 100, alpha=0.90, linewidths=0.1)
                        column.contour(dz[:, y_slice, :], 100, alpha=0.90, linewidths=0.1)
                    elif c == 4:
                        column.contour(dx[:, :, z_slice], 100, alpha=0.90, linewidths=0.1)
                        column.contour(dy[:, :, z_slice], 100, alpha=0.90, linewidths=0.1)

                    column.grid(False)
                    column.axis('off')
                    column.set_xticks([])
                    column.set_yticks([])
                    continue

            if c == 2 or c == 3 or c == 4 or c == 5:  # input images and deformed source + error map
                if input_image.shape[-1] == 1 and r > 0:
                    column.grid(False)
                    column.axis('off')
                    column.set_xticks([])
                    column.set_yticks([])
                    continue
            img = pred_segmentation_source[..., r] if c == 0 \
                else gt_segmentation_source[..., r] if c == 1 \
                else input_image[..., r] if c == 2 \
                else prediction[..., r] if c == 3 \
                else deformed_source[..., r] if c == 4 \
                else ground_truth[..., r] if c == 5 \
                else pred_segmentation_target[..., r] if c == 6 \
                else gt_segmentation_target[..., r] if c == 7 \
                else nontumor_mask[..., 0]

            img = img[x_slice, ...]  # select only 1 middle slice
            if c == 0:
                vmin = np.min(img)
                vmax = np.max(img)

            if c == 2 or c == 4 or c == 5:
                column.imshow(img, **kwargs)

            elif c == 3:

                column.imshow(img, cmap=cm.get_cmap('Reds'), vmin=0., vmax=3.)
                if nontumor_mask is not None:
                    mask = np.ma.masked_where(nontumor_mask[..., 0][x_slice, ...] < 0.9,
                                              nontumor_mask[..., 0][x_slice, ...])
                    column.imshow(mask, cmap='Greens', interpolation='none', vmin=0., vmax=2.)
            else:
                column.imshow(img, vmin=0., vmax=1., **kwargs)
            if r == 0:
                column.set_title(column_name)
            column.grid(False)
            column.axis('off')
            column.set_xticks([])
            column.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    plt.close()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def plot_results_for_registration(source, target, deformed_source, registration_map, registration_error,
                                  only_t1):
    kwargs = {'cmap': 'gray'}

    x_slice = int(source.shape[0] // 2)
    y_slice = int(source.shape[1] // 2)
    z_slice = int(source.shape[2] // 2)
    
    if only_t1:
        nb_row = 1
    else:
        nb_row = 4
        
    fig, ax = plt.subplots(nb_row, 5, squeeze=False)

    columns_names = ['Source', '(R(S)-T)^2', 'R(S)', 'Target', 'R']
    for r, row in enumerate(ax):
        for c, (column, column_name) in enumerate(zip(row, columns_names)):
            if c == 4:
                # Reformat registration grid
                dx, dy, dz = (registration_map[..., 0], registration_map[..., 1], registration_map[..., 2])
                if r == 0:
                    column.contour(dy[x_slice, :, :], 100, alpha=0.90, linewidths=0.1)
                    column.contour(dz[x_slice, :, :], 100, alpha=0.90, linewidths=0.1)
                elif r == 1:
                    column.contour(dx[:, y_slice, :], 100, alpha=0.90, linewidths=0.1)
                    column.contour(dz[:, y_slice, :], 100, alpha=0.90, linewidths=0.1)
                elif r == 2:
                    column.contour(dx[:, :, z_slice], 100, alpha=0.90, linewidths=0.1)
                    column.contour(dy[:, :, z_slice], 100, alpha=0.90, linewidths=0.1)

                column.grid(False)
                column.axis('off')
                column.set_xticks([])
                column.set_yticks([])
                continue

            img = source[..., r] if c == 0 \
                else registration_error[..., r] if c == 1 \
                else deformed_source[..., r] if c == 2 \
                else target[..., r]

            img = img[x_slice, ...]  # select only 1 middle slice
            # img = ndimage.rotate(img, 90)
            if c in [0, 2, 3]:
                column.imshow(img, **kwargs)
            else:
                column.imshow(img, cmap=cm.get_cmap('Reds'), vmin=0., vmax=3.)
            if r == 0:
                column.set_title(column_name)
            column.grid(False)
            column.axis('off')
            column.set_xticks([])
            column.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    plt.close()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """

    width, height, _ = tensor.shape

    image = Image.frombytes("RGBA", (width, height), tensor.tostring())
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height, width=width, colorspace=3, encoded_image_string=image_string)


class TensorBoardImage(keras.callbacks.Callback):

    def __init__(self, validation_generator,
                 log_path,
                 only_brats):

        super().__init__()
        self.validation_generator = validation_generator
        self.log_path = log_path
        self.only_brats = only_brats

    def on_epoch_end(self, epoch, logs={}):

        # Load image
        X, _ = self.validation_generator.__getitem__(0)

        if self.only_brats:
            moving, moving_mask, reference, reference_mask = X
        else:
            moving, moving_mask, reference = X

        # Do something to the image
        y = self.model.predict(X)

        deformed, displacements, deformed_mask = y

        batch_size = deformed.shape[0]

        writer = tf.summary.FileWriter(self.log_path)

        for batch in range(batch_size):
            tensor = plot_result(
                moving, reference, deformed, displacements, batch)

            tensor = make_image(tensor)

            summary = tf.Summary(
                value=[tf.Summary.Value(tag='Validation : ' + str(batch),
                                        image=tensor)])

            writer.add_summary(summary, epoch)

        writer.close()

        return


class TensorBoardImageSegmentation(keras.callbacks.Callback):
    def __init__(self, data_generator, log_path, set_name):
        super().__init__()
        self.data_generator = data_generator
        self.log_path = log_path
        self.set_name = set_name
        self.n_calls = 4

    def on_epoch_end(self, epoch, logs=None):
        writer = tf.summary.FileWriter(self.log_path)

        for k in range(self.n_calls):
            # Load data
            image_batch, ground_truth_batch = self.data_generator.__getitem__(k)

            # Do something to the image
            preds = self.model.predict(image_batch)
            for i, (input_img, pred, gt) in enumerate(zip(image_batch, preds, ground_truth_batch)):
                tensor = make_image(plot_results_for_segmentation(input_img, pred, gt))
                summary = tf.Summary(value=[tf.Summary.Value(tag=self.set_name + str(self.n_calls * k + i),
                                                             image=tensor)])
                writer.add_summary(summary, epoch)

        return writer.close()


class TensorBoardImageRegistrationSegmentation(keras.callbacks.Callback):
    def __init__(self, data_generator, log_path, set_name, decoder_segmentation_keys):
        super().__init__()
        self.data_generator = data_generator
        self.log_path = log_path
        self.set_name = set_name
        self.n_calls = 1

        self.decoder_segmentation_keys = decoder_segmentation_keys

    def on_epoch_end(self, epoch, logs=None):
        writer = tf.summary.FileWriter(self.log_path)

        for k in range(self.n_calls):
            # Load data
            input_data, label_data = self.data_generator.__getitem__(k)
            batch_source, batch_target = input_data
            batch_registration = label_data['registration_map_application']
            
            batch_gt_source_segmentation = label_data[self.decoder_segmentation_keys[0]]
            print('shape', batch_gt_source_segmentation.shape)
            batch_gt_target_segmentation = label_data[self.decoder_segmentation_keys[1]]


            # Do something to the image
            predictions = self.model.predict([batch_source, batch_target])

            pred_segmentation_source = predictions[0]
            pred_registration_map = predictions[1]
            pred_deformed_source = predictions[2]
            pred_segmentation_target = predictions[-1]

            print('len(predictions)', len(predictions))
            if len(predictions) == 6:  # with_loss_trick
                predicted_tumor_masked_deformed_source_minus_target = predictions[3]
                pred_nontumor_mask = predictions[4]
                for i, (source, deformed_source, target, nontumor_mask, pred_mask_source, pred_mask_target,
                        source_seg_gt, target_seg_gt, registration_map, registration_error) in enumerate(
                    zip(batch_source, pred_deformed_source, batch_registration, pred_nontumor_mask,
                        pred_segmentation_source, pred_segmentation_target, batch_gt_source_segmentation,
                        batch_gt_target_segmentation, pred_registration_map,
                        predicted_tumor_masked_deformed_source_minus_target)):
                    tensor = make_image(
                        plot_results_for_segmentation_registration(source, np.square(registration_error), target,
                                                                   invert_masks=True,
                                                                   nontumor_mask=nontumor_mask,
                                                                   pred_segmentation_source=pred_mask_source,
                                                                   pred_segmentation_target=pred_mask_target,
                                                                   gt_segmentation_source=source_seg_gt,
                                                                   gt_segmentation_target=target_seg_gt,
                                                                   deformed_source=deformed_source,
                                                                   registration_map=registration_map))
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag=self.set_name + str(self.n_calls * k + i), image=tensor)])
                    writer.add_summary(summary, epoch)
            else:

                for i, (source, deformed_source, target, pred_mask_source, pred_mask_target,
                        source_seg_gt, target_seg_gt, registration_map) in enumerate(
                    zip(batch_source, pred_deformed_source, batch_registration,
                        pred_segmentation_source, pred_segmentation_target, batch_gt_source_segmentation,
                        batch_gt_target_segmentation, pred_registration_map)):
                    tensor = make_image(
                        plot_results_for_segmentation_registration(source,
                                                                   np.square(deformed_source - target),
                                                                   target,
                                                                   invert_masks=True,
                                                                   nontumor_mask=None,
                                                                   pred_segmentation_source=pred_mask_source,
                                                                   pred_segmentation_target=pred_mask_target,
                                                                   gt_segmentation_source=source_seg_gt,
                                                                   gt_segmentation_target=target_seg_gt,
                                                                   deformed_source=deformed_source,
                                                                   registration_map=registration_map))
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag=self.set_name + str(self.n_calls * k + i), image=tensor)])
                    writer.add_summary(summary, epoch)

        return writer.close()


class TensorBoardImageRegistration(keras.callbacks.Callback):
    def __init__(self, data_generator, log_path, set_name, only_t1):
        super().__init__()
        self.data_generator = data_generator
        self.log_path = log_path
        self.set_name = set_name
        self.n_calls = 4
        self.only_t1 = only_t1
        
    def on_epoch_end(self, epoch, logs=None):
        writer = tf.summary.FileWriter(self.log_path)

        for k in range(self.n_calls):
            # Load data
            input_data, label_data = self.data_generator.__getitem__(k)
            batch_source, batch_target = input_data
            batch_registration = label_data['registration_map_application']

            # Do something to the image
            predictions = self.model.predict([batch_source, batch_target])

            pred_registration_map = predictions[0]
            pred_deformed_source = predictions[1]

            for i, (source, deformed_source, target, registration_map) in enumerate(
                    zip(batch_source, pred_deformed_source, batch_registration, pred_registration_map)):
                tensor = make_image(
                    plot_results_for_registration(source, target, deformed_source=deformed_source,
                                                  registration_map=registration_map,
                                                  registration_error=np.square(deformed_source
                                                                               -
                                                                               target),
                                                  only_t1=self.only_t1))
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=self.set_name + str(self.n_calls * k + i), image=tensor)])
                writer.add_summary(summary, epoch)

        return writer.close()
