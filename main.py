#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:29:50 2019

@author: theoestienne
"""
import os
import sys
import argparse
import keras.utils
import keras.callbacks as callbacks
import keras.losses
import keras.optimizers as optimizers
import keras.metrics
import keras.regularizers
import functools
import math
import numpy as np
import time
# My package
from joint_registration_tumor_segmentation import model_loader
from joint_registration_tumor_segmentation import Dataset
from joint_registration_tumor_segmentation import ImageTensorboard
from joint_registration_tumor_segmentation import losses
from joint_registration_tumor_segmentation.tools import log

main_path = os.path.abspath(__file__)
n = main_path.find('Python')
if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)

n_output_channels = 4  # 4 classes
n_input_channels = 4  # 4 modalities


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Keras automatic registration', add_help=add_help)

    parser.add_argument('--segmentation-only', default=False, action='store_true',
                        help='True to train a Vnet on segmentation task')
    parser.add_argument('--only-t1', action='store_true', help='True to use only T1 in MRI sequences')
    parser.add_argument('--registration-only', default=False, action='store_true',
                        help='True to train a Vnet on registration task')
    parser.add_argument('--with-loss-trick', action='store_true',
                        help='For registration+segmentation only: use loss trick that mask the loss maps with '
                             'the predicted tumors (all classes except background)')
    parser.add_argument('--source-target-merge-operation', default='subtraction', type=str,
                        help='Elementiwse operation that merges vnet codes of source and target; supported: '
                             '"subtraction", "addition", "concatenation"')
    parser.add_argument('--ratio-weights-registration-over-segmentation', default=1., type=float,
                        help='Registration loss wil have weights of 1.; accordingly, put weight of 1./(2*ARG) for each '
                             'decoder loss.')
    parser.add_argument('--gpu', '-g', default=0, type=int, metavar='N',
                        help='Index of GPU used for calcul (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')
    parser.add_argument('--n-channels-first-layer', type=int, default=8,
                        help='Number of channels of first layer of vnet-style nets ')
    parser.add_argument('--batch-size', '-b', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--patience', default=15, type=int, metavar='N_EPOCHS', help='patience for early stopping')
    parser.add_argument('--lr', '--learning-rate', default=.001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--tensorboard', default=True, action='store_false', help='use tensorboard_logger to save data')
    parser.add_argument('--image-tensorboard', default=True, action='store_false',
                        help='use image tensorboard_logger to save image')
    parser.add_argument('--parallel', action='store_false', default=True, help='Use data parallel in CUDA')
    parser.add_argument('--nb-gpu', default=0, type=int, metavar='N',
                        help='Number of gpu in case of parallel calculation')
    parser.add_argument('--save', '-s', action='store_false', default=True, help='Save the model during training')
    parser.add_argument('--workers', '-w', default=4, type=int, help='Use multiprocessing for dataloader')
    parser.add_argument('--use-affine', action='store_false', default=True,
                        help='Use affine transformation in addition to deformable deformation')
    parser.add_argument('--deform-reg', type=float, default=1e-11, help='Regularisation of the deformation layer')
    parser.add_argument('--crop-size', type=float, nargs='+', default=240)
    parser.add_argument('--early-stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--session-name', type=str, default='', help='Give a name to the session')
    parser.add_argument('--lr-decrease', action='store_true', help='Reduce the learning rate')
    parser.add_argument('--lr-epochs-drop', type=int, default=20,
                        help='Number of epochs before decreasing the learning rate')
    parser.add_argument('--lr-drop', type=float, default=0.5, help='Drop factor of the learning rate')
    parser.add_argument('--create-new-split', action='store_true', help='Create a train, validation, test split')
    parser.add_argument('--translation', action='store_true', default=False, help='Do translation on dataset')
    parser.add_argument('--inference-path', type=str, help='True to perform inference on validation set')
    parser.add_argument('--load-path', type=str, help='Name of model loaded')
    parser.add_argument('--cohort', type=str, default='brats', choices=['oasis', 'brats'], help='Cohort used')
    parser.add_argument('--concatenated-input-arch', default=False, action='store_true',
                        help='True to consider input as the merged channelwise of source and target ie without shared encoder/decoder')

    return parser


# learning rate schedule
def step_decay(epoch, args):
    initial_lrate = args.lr
    drop = args.lr_drop
    epochs_drop = args.lr_epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate

def main(args):
    save_path = main_path + 'save/'
    session_name = args.session_name + '_' + time.strftime('%m.%d-%Hh%M')

    # Log
    log_path = save_path + 'training_log/' + session_name + '.log'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logger = log.set_logger(log_path)
    # trick to log the print functions (also prints on console since there is a console handler in logger)
    logger.write = logger.info
    logger.flush = lambda: None
    sys.stdout = logger

    if args.registration_only and args.segmentation_only:
        logger.error('Called with --segmentation-only and --registration-only: not possible')
        exit(-1)

    data_path = main_path + 'data/'
    if args.inference_path:
        brats_path = args.inference_path
    else:
        brats_path = data_path + 'BRATS/numpy/'
    oasis_path = data_path + 'oasis/'
    dataset_path = main_path + 'datset/'

    # logs some path info and arguments
    logger.info('save_path ' + save_path)
    logger.info('data_path ' + data_path)
    logger.info('brats_path ' + brats_path)
    logger.info('oasis_path ' + oasis_path)
    logger.info('Arguments: ' + ', '.join(['{}: {}'.format(arg, value) for arg, value in sorted(vars(args).items())]))
    logger.info('Original command line: {}'.format(' '.join(sys.argv)))

    args.model_path = main_path + 'save/models/'

    if args.cohort == 'oasis':
        args.crop_size = (160, 176, 208)
    else:
        args.crop_size = (144, 208, 144)

    # DataGen Parameters
    params = {'data_path': data_path, 'dim': args.crop_size, 'batch_size': args.batch_size, 'shuffle': True,
              'translation': args.translation, 'cohort': args.cohort,
              'mono_gpu': args.nb_gpu <= 1,
              'concatenate_inputs': args.concatenated_input_arch}

    logger.info('DataGen parameters: {}'.format(params))

    # Datasets
    if not args.inference_path:
        if args.create_new_split:
            brats_files = Dataset.load_datasets(brats_path, 'brats')
            oasis_files = Dataset.load_datasets(oasis_path, 'oasis')

            brats_files_train, brats_files_validation, brats_files_test = Dataset.create_dataset(brats_files,
                                                                                                 dataset_path, 'brats')
            oasis_files_train, oasis_files_validation, oasis_files_test = \
                    Dataset.create_dataset(oasis_files, dataset_path, 'oasis')
        else:
            brats_files_train, brats_files_validation, brats_files_test = Dataset.load_existing_dataset(dataset_path, 'brats')
            oasis_files_train, oasis_files_validation, oasis_files_test = Dataset.load_existing_dataset(dataset_path, 'oasis')

    if args.only_t1:
        global n_input_channels
        n_input_channels = 1

    # Generators
    DataGen = Dataset.DataGenerator

    mono_patient = True if args.segmentation_only else False

    # ground truths masks need to be unravel on last channel for some losses (eg categorical CE)
    to_categorical = n_output_channels >= 2

    if args.inference_path:
        inference_files_ids = os.listdir(args.inference_path)
        params['data_path'] = args.inference_path
        params['batch_size'] = 1
        inference_generator = DataGen(inference_files_ids, n_input_channels=n_input_channels,
                                      n_output_channels=n_output_channels, validation=True,
                                      mono_patient=mono_patient, to_categorical=to_categorical,
                                      inference=True, **params)
    else:
        if args.cohort == 'oasis':
            (files_train,
             files_validation, files_test) = (oasis_files_train, oasis_files_validation,
                                              oasis_files_test)
        elif args.cohort == 'brats':
            (files_train,
             files_validation, files_test) = (brats_files_train, brats_files_validation,
                                              brats_files_test)

        training_generator = DataGen(files_train, n_input_channels=n_input_channels,
                                     n_output_channels=n_output_channels, mono_patient=mono_patient,
                                     to_categorical=to_categorical, **params)
        validation_generator = DataGen(files_validation, n_input_channels=n_input_channels,
                                       n_output_channels=n_output_channels, validation=True,
                                       mono_patient=mono_patient, to_categorical=to_categorical, **params)

    # Design model
    if args.concatenated_input_arch:
        with_source_segmenter = False
        with_target_segmenter = False
        args.registration_only = True

        model = model_loader.conc_input_handwritten_vnet((*args.crop_size, n_input_channels),
                                                         'softmax', n_output_channels=n_output_channels,
                                                         filter_width_normal_conv=3,
                                                         first_conv_n_filters=args.n_channels_first_layer,
                                                         deform_regularisation=args.deform_reg)

    elif args.segmentation_only:
        model, _, _ = model_loader.handwritten_vnet((*args.crop_size, n_input_channels),
                                                    'softmax', n_output_channels=n_output_channels,
                                                    filter_width_normal_conv=3,
                                                    first_conv_n_filters=args.n_channels_first_layer)
    else:
        if args.registration_only:
            with_source_segmenter = False
            with_target_segmenter = False
        else:
            with_source_segmenter = True
            with_target_segmenter = True
        model, _, _ = model_loader.get_registration_segmenter_vnet((*args.crop_size, n_input_channels),
                                                                   'softmax', n_output_channels=n_output_channels,
                                                                   filter_width_normal_conv=3,
                                                                   first_conv_n_filters=args.n_channels_first_layer,
                                                                   with_source_segmenter=with_source_segmenter,
                                                                   with_target_segmenter=with_target_segmenter,
                                                                   registration_activity_l2reg_coef=args.L2_loss_coeff,
                                                                   source_target_merge_operation=args.source_target_merge_operation,
                                                                   with_loss_trick=args.with_loss_trick,
                                                                   deform_regularisation=args.deform_reg)

    # Perform inference if specified (don't need to define any loss, optimizer or such)
    if args.inference_path:
        print('before', model.layers[2].get_weights()[1])
        assert args.load_path is not None, 'performing inference, expected model path, found None'
        model.load_weights(args.load_path, by_name=True)
        print('after', model.layers[2].get_weights()[1])
        predictions = model.predict_generator(inference_generator, verbose=1)
        output_dir = 'predictions_segmentation'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for p, p_id in zip(predictions, inference_files_ids):
            np.save(os.path.join(output_dir, p_id), p)
        print(len(predictions))
        print([p.shape for p in predictions])
        exit()

    # Callbacks
    kwargs = {}
    callbacks_list = []
    if args.tensorboard:
        log_path = save_path + 'logs/' + session_name + '/'
        tensorboard = callbacks.TensorBoard(log_dir=log_path, update_freq='batch', histogram_freq=0)

        if args.image_tensorboard:
            if args.segmentation_only:
                tensorboard_image_class = ImageTensorboard.TensorBoardImageSegmentation
            elif args.registration_only:
                tensorboard_image_class = ImageTensorboard.TensorBoardImageRegistration
                kwargs['only_t1'] = args.only_t1
            else:
                tensorboard_image_class = ImageTensorboard.TensorBoardImageRegistrationSegmentation
                kwargs['decoder_segmentation_keys'] = ['decoder_segmentation_1', 'decoder_segmentation_2']

            # add validation and training images plots to tensorboard
            tensorboard_image_val = tensorboard_image_class(
                data_generator=validation_generator, log_path=log_path, set_name='validation', **kwargs)
            tensorboard_image_train = tensorboard_image_class(
                data_generator=training_generator, log_path=log_path, set_name='training', **kwargs)
            callbacks_list.extend([tensorboard, tensorboard_image_val, tensorboard_image_train])
        else:
            callbacks_list.extend([tensorboard])

    if args.save:
        model_path = save_path + 'models/' + session_name + '/'
        print('model_path', model_path)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        save_callback = callbacks.ModelCheckpoint(model_path + 'model.{epoch:03d}--{val_loss:.3f}.h5',
                                                  save_weights_only=False, period=5)
        callbacks_list.append(save_callback)

    if args.early_stopping:
        callbacks_list.append(callbacks.EarlyStopping(patience=args.patience, min_delta=.001))
    if args.lr_decrease:
        callbacks_list.append(callbacks.LearningRateScheduler(functools.partial(step_decay, args=args)))


    if args.load_path:
        model.load_weights(args.load_path, by_name=True)
        logger.error('LOADED MODEL WEIGHTS {}'.format(args.load_path))

    print('summary', model.summary(200))
    if args.parallel and args.nb_gpu > 1:
        model = keras.utils.multi_gpu_model(model, gpus=args.nb_gpu)


    # Optimizer and compile scheme
    optim = optimizers.Adam(lr=args.lr)

    #### Loss and metrics

    segmentation_metrics = losses.averaged_dice_loss
    segmentation_loss = 'categorical_crossentropy'

    registration_map_construction_loss = losses.zero_loss  # no reg yet
    registration_map_construction_metrics = losses.mean_squared_error_with_zero
    registration_map_application_losstrick_loss = losses.mean_squared_error_with_zero  # output is masked(deformed(source)-target)

    if args.with_loss_trick:
        registration_map_application_loss = losses.zero_loss  # output is deformed(source), before masking, so do nothing
    else:
        registration_map_application_loss = 'mean_squared_error'  # output is deformed(source), no mask; with target

    if args.segmentation_only:
        model.compile(optimizer=optim, loss=segmentation_loss,
                      metrics=[segmentation_metrics])
    else:
        loss_dict = {'decoder_segmentation_1': segmentation_loss,
                     'decoder_segmentation_2': segmentation_loss,
                     'registration_map_construction': registration_map_construction_loss,
                     'registration_map_application': registration_map_application_loss,  # used without loss trick
                     'registration_map_application_losstrick':
                         registration_map_application_losstrick_loss,  # with loss trick
                     'predicted_background_merged_mask': losses.zero_loss,  # do not do anything with extracted tumor masks
                     }
        metrics_dict = {'decoder_segmentation_1': segmentation_metrics,
                        'decoder_segmentation_2': segmentation_metrics,
                        'registration_map_construction': registration_map_construction_metrics,
                        'registration_map_application': 'mean_squared_error',
                        'registration_map_application_losstrick': losses.zero_loss,
                        'predicted_background_merged_mask': losses.zero_loss
                        }
        reg_over_seg_ratio = args.ratio_weights_registration_over_segmentation
        decoder_weight = 1. / (2. * float(reg_over_seg_ratio))
        loss_weights_dict = {'decoder_segmentation_1': decoder_weight,
                             'decoder_segmentation_2': decoder_weight,
                             'registration_map_construction': 0.,
                             'registration_map_application': 1.,
                             'registration_map_application_losstrick': 1.,
                             'predicted_background_merged_mask': 0.,
                             }

        keys = ['registration_map_application', 'registration_map_construction']
        if with_source_segmenter:
            keys.append('decoder_segmentation_1')
        if with_target_segmenter:
            keys.append('decoder_segmentation_2')
        if args.with_loss_trick:
            keys.append('predicted_background_merged_mask')
            keys.append('registration_map_application_losstrick')
        loss_dict = {k: v for k, v in loss_dict.items() if k in keys}
        metrics_dict = {k: v for k, v in metrics_dict.items() if k in keys}
        loss_weights_dict = {k: v for k, v in loss_weights_dict.items() if k in keys}
        model.compile(optimizer=optim, loss=loss_dict, metrics=metrics_dict,
                      loss_weights=loss_weights_dict)

    # train and evaluate on both validation and test sets
    train_history = train(model, training_generator, validation_generator, callbacks_list, args)
    logger.info('Training history: {}'.format(train_history.history))


def train(model, training_generator, validation_generator, callbacks, args):
    fit_kwargs = {'epochs': args.epochs, 'callbacks': callbacks, 'workers': args.workers}

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  # steps_per_epoch=1,
                                  validation_data=validation_generator,
                                  validation_steps=len(validation_generator), **fit_kwargs)
    return history


def evaluate(model, test_generator, args):
    history = model.evaluate_generator(generator=test_generator, workers=1, verbose=1)
    return history


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    main(args)
