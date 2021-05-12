# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:21:38 2019

@author: T_ESTIENNE
"""
import os
import sys
import datetime
import argparse
import numpy as np
import keras.layers as layers
import keras.models as models
from tqdm import tqdm

# My package
from joint_registration_tumor_segmentation import Dataset
from joint_registration_tumor_segmentation.tools import log
from joint_registration_tumor_segmentation.tools import numpy2nifti
from joint_registration_tumor_segmentation.tools.utils import load_model_v2
from joint_registration_tumor_segmentation.diffeomorphicTransformer import diffeomorphicTransformer3D
import tensorflow as tf

save_numpy_predictions = False

tf.logging.set_verbosity(tf.logging.DEBUG)

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

save_path = main_path + 'save_inference/frontiers_brain/'

# Log
log_path = save_path + 'inference_log/' + str(datetime.datetime) + '.log'
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
logger = log.set_logger(log_path)
# trick to log the print functions (also prints on console since there is a console handler in logger)
logger.write = logger.info
logger.flush = lambda: None
sys.stdout = logger


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='Keras automatic registration', add_help=add_help)

    parser.add_argument('--data-folder-path', default=None, type=str,
                        help='Parent folder containing numpy images to infer')
    parser.add_argument('--cohort', type=str, default='brats', choices=['oasis', 'brats'], help='Cohort used')
    parser.add_argument('--output-folder', default=None, type=str, help='Folder in which to put predictions')

    parser.add_argument('--segmentation-only', default=False, action='store_true',
                        help='True to train a Vnet on segmentation task')
    parser.add_argument('--registration-only', default=False, action='store_true',
                        help='True to train a Vnet on registration task')
    parser.add_argument('--model-abspath', default=None, type=str,
                        help='Absolute path of the model with which to perform inference; overrides '
                             '--segmentation-only and --registration-only')
    parser.add_argument('--models-folder', default=None, type=str,
                        help='Absolute path of a folder containing source saved models.')
    parser.add_argument('--rigid', action='store_true', default=False,
                        help='Toggle inference for translation only model.')

    parser.add_argument('--with-loss-trick', action='store_true',
                        help='For registration+segmentation only: use loss trick that mask the loss maps with '
                             'the predicted tumors (all classes except background)')
    parser.add_argument('--source-target-merge-operation', default='subtraction', type=str,
                        help='Elementiwse operation that merges vnet codes of source and target; supported: '
                             '"subtraction", "addition", "concatenation"')
    parser.add_argument('--translation', action='store_true', default=False, help='Do translation on dataset')

    parser.add_argument('--get-segmentation', action='store_true', default=False,
                        help='True to compute pred segmentation: return masks')
    parser.add_argument('--get-registration', action='store_true', default=False,
                        help='True to compute pred registration: return deformed source')
    parser.add_argument('--only-t1', action='store_true', help='True to use only T1 in MRI sequences')
    parser.add_argument('--deformed-brats-gt-mask', action='store_true',
                        help='Deformed brats ground truth mask for volume calculation')
    parser.add_argument('--no-layer6', action='store_true', default=False,
                        help='Do not take layer 6')
    return parser


def deformed_mask_model(dim, mask_channel):
    moving_mask = layers.Input((*dim, mask_channel))
    grid = layers.Input((*dim, 3))

    deformed_mask = diffeomorphicTransformer3D()([moving_mask, grid])
    model = models.Model(input=[moving_mask, grid],
                         output=[deformed_mask])

    model.compile(optimizer='sgd', loss='mse')

    return model


def main(args):
    logger.info('Performing inference on cohort {}'.format(args.cohort.lower()))

    if args.output_folder:
        output_dir = args.output_folder
    else:
        output_dir = args.model_abspath[:-3] + '/'

    # output_dir = args.output_folder if args.output_folder else 'predictions_segmentation'

    # assert args.get_segmentation or args.get_registration, 'Select either --get-segmentation or --get-registration'
    if args.get_segmentation and args.get_registration:
        logger.error('Select either --get-segmentation or --get-registration but not both; '
                     'if you need both, call twice this script with either')
        exit(-1)

    if args.registration_only and args.segmentation_only:
        logger.error('Called with --segmentation-only and --registration-only: not possible')
        exit(-1)

    if args.get_segmentation and args.registration_only:
        logger.error('Calling --get-segmentation with --registration-only: uncompatible')
        exit(-1)
    if args.get_registration and args.segmentation_only:
        logger.error('Calling --get-registration with --segmentation-only: uncompatible')
        exit(-1)

    if args.data_folder_path is None:
        raise ValueError('Found data folder path of value None, '
                         'please specify parent folder of numpy arrays in --data-folder-path')

    if args.deformed_brats_gt_mask and args.cohort.lower() != 'brats':
        raise ValueError('If you want to deformed brats GT mask, cohort must be equal to brats')

    if args.deformed_brats_gt_mask and args.segmentation_only:
        raise ValueError('Calling --deformed-brats-gt-mask with --segmentation-only : uncompatible')

    if args.cohort.lower() == 'oasis' and not args.translation:
        raise ValueError('With oasis cohort, translation should be true')

    data_path = os.path.abspath(args.data_folder_path) + '/'

    # logs some path info and arguments
    logger.info('data_path ' + data_path)
    logger.info('Arguments: ' + ', '.join(['{}: {}'.format(arg, value) for arg, value in sorted(vars(args).items())]))
    logger.info('Original command line: {}'.format(' '.join(sys.argv)))

    crop_size = (144, 208, 144) if args.cohort.lower() in ['brats', 'oasis'] else (160, 176, 208)

    # DataGen Parameters
    params = {'data_path': data_path,
              'dim': crop_size,
              'batch_size': 1,
              'shuffle': False,
              'translation': args.translation,
              'only_close_tumor': False,
              'close_tumor_nb': -1,
              'cohort': args.cohort,
              'concatenate_inputs': 'ConcInputs' in args.model_abspath}
    logger.info('DataGen parameters: {}'.format(params))

    # Generators
    DataGen = Dataset.DataGenerator

    mono_patient = True if args.segmentation_only else False
    # ground truths masks need to be unravel on last channel for some losses (eg categorical CE)

    n_output_channels = 4  # 4 classes
    n_input_channels = 1 if args.only_t1 else 4  # 4 modalities
    to_categorical = n_output_channels >= 2
    debug = False
    inference = not args.deformed_brats_gt_mask

    if args.cohort.lower() == 'oasis':
        if debug:
            inference_files_ids = [file for file in os.listdir(args.data_folder_path)
                                   if 'orig' in file]
            inference_files_ids = inference_files_ids[:6]  ### TEST
        else:
            inference_files_ids = np.loadtxt(args.data_folder_path + 'test.txt',
                                             dtype=str)
        to_categorical = False
        mask_channel = 29
        n_output_channels = 29
        n_input_channels = 1
        original_images = [Dataset.load_mgh(args.data_folder_path + file) for file in inference_files_ids]
        names_images = inference_files_ids

    elif args.deformed_brats_gt_mask:

        mask_channel = 4

        inference_files_ids = os.listdir(os.path.join(args.data_folder_path, 'BRATS/numpy'))[:3]
        original_images = [np.load(args.data_folder_path + 'BRATS/numpy/' + file) for file in inference_files_ids]

        names_images = inference_files_ids

    elif args.cohort.lower() == 'brats':

        inference_files_ids = os.listdir(args.data_folder_path)
        original_images = [np.load(os.path.join(args.data_folder_path, f))
                           for f in os.listdir(args.data_folder_path) if f.endswith('.npy')]

        names_images = os.listdir(args.data_folder_path)

    else:
        raise ValueError('Cohort should be equal to brats or oasis')

    inference_generator = DataGen(inference_files_ids, n_input_channels=n_input_channels,
                                  n_output_channels=n_output_channels, validation=True,
                                  mono_patient=mono_patient, to_categorical=to_categorical,
                                  inference=inference, **params)

    if args.rigid:
        assert args.translation, 'Rigid should be used with --translation for rigorous inference'
        class Object(object):
            pass
        model = Object()
        model.predict_on_batch = lambda x: x  # identity since
    else:
        # Retrieve trained weights
        if args.model_abspath is not None:
            model_path = args.model_abspath
        else:
            raise ValueError

        # Load pretrained model
        model = load_model_v2(model_path)

        if not args.no_layer6:
            a = model.layers[6]
            model = a
    # Perform inference

    # Save results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if args.deformed_brats_gt_mask:
            os.makedirs(output_dir + '/HGG/')
            os.makedirs(output_dir + '/LGG/')

    print('n original images', len(original_images), 'shape original images', original_images[0].shape)

    # Predict result
    iter_generator = iter(inference_generator)

    if args.rigid:
        grid_model = Object()
        grid_model.predict_on_batch = lambda x: x
    else:
        grid_model = deformed_mask_model(crop_size, mask_channel)

    for i in tqdm(range(len(inference_generator))):

        X, y = next(iter_generator)

        predictions = model.predict_on_batch(X)
        if len(predictions) == 4:  # reg+seg without loss trick

            decoded_sources = predictions[0]
            registration_maps = predictions[1]
            deformed_sources = predictions[2]
            predicted_tumor_masked_deformed_source_minus_targets = []
            predicted_nontumor_masks = []
            decoded_targets = predictions[3]
        elif len(predictions) == 2:

            registration_maps = predictions[0]
            deformed_sources = predictions[1]
            decoded_sources = []
            predicted_tumor_masked_deformed_source_minus_targets = []
            decoded_targets = []

        elif len(predictions) == 6:  # reg+seg with loss trick

            decoded_sources = predictions[0]
            registration_maps = predictions[1]
            deformed_sources = predictions[2]
            predicted_tumor_masked_deformed_source_minus_targets = predictions[3]
            predicted_nontumor_masks = predictions[4]
            decoded_targets = predictions[5]
        else:
            raise ValueError

        if args.cohort == 'oasis' or args.deformed_brats_gt_mask:
            mask_source = y['decoder_segmentation_1']

            test_input = [mask_source, registration_maps]

            deformed_mask = grid_model.predict_on_batch(test_input)
            print('mask_source', np.sum(mask_source != 0))
            print('deformed_mask', np.sum(deformed_mask != 0))
            deformed_mask = [np.squeeze(deformed_mask)]
        else:
            deformed_mask = []

        current_names = [names_images[i]]
        containers = [decoded_sources, deformed_sources, registration_maps,
                      predicted_tumor_masked_deformed_source_minus_targets, deformed_mask]
        containers_names = ['decoded_source', 'deformed_source', 'registration_map', 'registration_error_map',
                            'deformed_mask']

        for predictions, image_type in zip(containers, containers_names):
            is_segmentation_mask = image_type in ['decoded_source', 'deformed_mask']

            numpy2nifti.convert_arrays_to_submission(names_images=current_names,
                                                     numpy_images=original_images, numpy_predictions=predictions,
                                                     output_folder=output_dir,
                                                     is_segmentation_mask=is_segmentation_mask,
                                                     filename_suffix_before_format=image_type)


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()

    if args.models_folder is not None:
        output_folder = str(args.output_folder) if args.output_folder is not None else str(args.models_folder)
        models_ = list(filter(lambda f: f.endswith('.h5'), os.listdir(args.models_folder)))
        assert len(models_) > 0

        for model_rel_path in models_:
            print('performing inference of model {}'.format(model_rel_path))
            args.model_abspath = os.path.join(args.models_folder, model_rel_path)
            args.output_folder = os.path.join(output_folder, model_rel_path)[:-3]
            args.only_t1 = '_onlyt1_' in model_rel_path
            args.no_layer6 = 'seg_only_' in model_rel_path or 'reg_only_' in model_rel_path
            if not os.path.exists(args.output_folder):
                main(args)
            else:
                print('Discarding {}: already exists'.format(args.output_folder))
    else:
        main(args)
