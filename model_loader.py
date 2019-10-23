# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:35:54 2019

@author: T_ESTIENNE
"""
import numpy as np
from keras import backend as K
import keras.models as models
from keras.layers import Input, Conv3D, Deconvolution3D, add, Concatenate, subtract, Lambda, multiply, concatenate, minimum
from keras.utils import plot_model
from frontiers_code import diffeomorphicTransformer as Transformer


def handwritten_vnet(input_shape, final_activation_name, n_output_channels, filter_width_normal_conv=5,
                     first_conv_n_filters=16, input_layer=None, activity_l2reg_coef=1e-3,
                     last_layer_zero_initialization=False, override_decoder_input_channels=None,
                     get_only_decoder=False, deform_regularisation=0):
    w = filter_width_normal_conv
    f = first_conv_n_filters

    if deform_regularisation > 0:
        last_activation_regularization = Transformer.DefReg(alpha=deform_regularisation,
                                                       value=0.5)
    else:
        last_activation_regularization = None

    inputs = encoder_inputs = Input(
        input_shape) if input_layer is None else input_layer
    encoder_reformatted_inputs = Conv3D(f, (1, 1, 1), strides=1, padding='same', name='reformatted_inputs')(
        encoder_inputs)

    # Encoder
    encoder_block1_conv1 = Conv3D(f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block1_conv1')(encoder_reformatted_inputs)
    encoder_block1_add = add(
        [encoder_block1_conv1, encoder_reformatted_inputs], name='encoder_block1_add')
    encoder_block1_downed = Conv3D(2 * f, (2, 2, 2), strides=2,
                                   padding='same', activation='relu',
                                   name='encoder_block1_downconv')(encoder_block1_add)

    encoder_block2_conv1 = Conv3D(2 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block2_conv1')(encoder_block1_downed)
    encoder_block2_conv2 = Conv3D(2 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block2_conv2')(encoder_block2_conv1)
    encoder_block2_add = add(
        [encoder_block1_downed, encoder_block2_conv2], name='encoder_block2_add')
    encoder_block2_downed = Conv3D(4 * f, (2, 2, 2), strides=2,
                                   padding='same', activation='relu',
                                   name='encoder_block2_downconv')(encoder_block2_add)

    encoder_block3_conv1 = Conv3D(4 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block3_conv1')(encoder_block2_downed)
    encoder_block3_conv2 = Conv3D(4 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block3_conv2')(encoder_block3_conv1)
    encoder_block3_conv3 = Conv3D(4 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block3_conv3')(encoder_block3_conv2)
    encoder_block3_add = add(
        [encoder_block2_downed, encoder_block3_conv3], name='encoder_block3_add')
    encoder_block3_downed = Conv3D(8 * f, (2, 2, 2), strides=2,
                                   padding='same', activation='relu',
                                   name='encoder_block3_downconv')(encoder_block3_add)

    encoder_block4_conv1 = Conv3D(8 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block4_conv1')(encoder_block3_downed)
    encoder_block4_conv2 = Conv3D(8 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block4_conv2')(encoder_block4_conv1)
    encoder_block4_conv3 = Conv3D(8 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block4_conv3')(encoder_block4_conv2)
    encoder_block4_add = add(
        [encoder_block3_downed, encoder_block4_conv3], name='encoder_block4_add')
    encoder_block4_downed = Conv3D(16 * f, (2, 2, 2), strides=2, padding='same', activation='relu',
                                   name='encoder_block4_downconv')(encoder_block4_add)

    encoder_block5_conv1 = Conv3D(16 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block5_conv1')(encoder_block4_downed)
    encoder_block5_conv2 = Conv3D(16 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block5_conv2')(encoder_block5_conv1)
    encoder_block5_conv3 = Conv3D(16 * f, (w, w, w), strides=1,
                                  padding='same', activation='relu',
                                  name='encoder_block5_conv3')(encoder_block5_conv2)
    encoder_block5_add = add(
        [encoder_block4_downed, encoder_block5_conv3], name='encoder_block5_add')

    vnet_encoder = models.Model(inputs=encoder_inputs, outputs=[encoder_block1_add, encoder_block2_add,
                                                                encoder_block3_add, encoder_block4_add,
                                                                encoder_block5_add])
    vnet_encoder.name = 'vnet_encoder'

    # Decoder
    # instantiate Input placeholders for keras: infer the number of channels in normal regime, or override with override_decoder_input_channels if it is not None
    if override_decoder_input_channels is None:
        decoder_input_encoderblock1 = Input(
            encoder_block1_add.get_shape().as_list()[1:])
        decoder_input_encoderblock2 = Input(
            encoder_block2_add.get_shape().as_list()[1:])
        decoder_input_encoderblock3 = Input(
            encoder_block3_add.get_shape().as_list()[1:])
        decoder_input_encoderblock4 = Input(
            encoder_block4_add.get_shape().as_list()[1:])
        decoder_input_encoderblock5 = Input(
            encoder_block5_add.get_shape().as_list()[1:])
    else:
        assert isinstance(override_decoder_input_channels, list)
        assert len(
            override_decoder_input_channels) == 5, 'expected 5 int for overriding channels since there are 5 res blocks'
        decoder_input_encoderblock1 = Input(
            encoder_block1_add.get_shape().as_list()[1:-1] + [override_decoder_input_channels[0]])
        decoder_input_encoderblock2 = Input(
            encoder_block2_add.get_shape().as_list()[1:-1] + [override_decoder_input_channels[1]])
        decoder_input_encoderblock3 = Input(
            encoder_block3_add.get_shape().as_list()[1:-1] + [override_decoder_input_channels[2]])
        decoder_input_encoderblock4 = Input(
            encoder_block4_add.get_shape().as_list()[1:-1] + [override_decoder_input_channels[3]])
        decoder_input_encoderblock5 = Input(
            encoder_block5_add.get_shape().as_list()[1:-1] + [override_decoder_input_channels[4]])

    decoder_block0_upped = Deconvolution3D(8 * f, (2, 2, 2), strides=2, padding='same', activation='relu',
                                           name='decoder_block0_upconv')(decoder_input_encoderblock5)
    decoder_block1_features_forwarding = Concatenate()(
        [decoder_input_encoderblock4, decoder_block0_upped])
    decoder_block1_conv1 = Conv3D(8 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block1_conv1')(decoder_block1_features_forwarding)
    decoder_block1_conv2 = Conv3D(8 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block1_conv2')(decoder_block1_conv1)
    decoder_block1_conv3 = Conv3D(8 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block1_conv3')(decoder_block1_conv2)
    decoder_block1_add = add(
        [decoder_block0_upped, decoder_block1_conv3], name='decoder_block1_add')

    decoder_block1_upped = Deconvolution3D(4 * f, (2, 2, 2), strides=2, padding='same', activation='relu',
                                           name='decoder_block1_upconv')(decoder_block1_add)
    decoder_block2_features_forwarding = Concatenate()(
        [decoder_input_encoderblock3, decoder_block1_upped])
    decoder_block2_conv1 = Conv3D(4 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block2_conv1')(decoder_block2_features_forwarding)
    decoder_block2_conv2 = Conv3D(4 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block2_conv2')(decoder_block2_conv1)
    decoder_block2_conv3 = Conv3D(4 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block2_conv3')(decoder_block2_conv2)
    decoder_block2_add = add(
        [decoder_block1_upped, decoder_block2_conv3], name='decoder_block2_add')

    decoder_block2_upped = Deconvolution3D(2 * f, (2, 2, 2), strides=2, padding='same', activation='relu',
                                           name='decoder_block2_upconv')(decoder_block2_add)
    decoder_block3_features_forwarding = Concatenate()(
        [decoder_input_encoderblock2, decoder_block2_upped])
    decoder_block3_conv1 = Conv3D(2 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block3_conv1')(decoder_block3_features_forwarding)
    decoder_block3_conv2 = Conv3D(2 * f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block3_conv2')(decoder_block3_conv1)
    decoder_block3_add = add(
        [decoder_block2_upped, decoder_block3_conv2], name='decoder_block3_add')

    decoder_block3_upped = Deconvolution3D(f, (2, 2, 2), strides=2, padding='same', activation='relu',
                                           name='decoder_block3_upconv')(decoder_block3_add)
    decoder_block4_features_forwarding = Concatenate()(
        [decoder_input_encoderblock1, decoder_block3_upped])
    decoder_block4_conv1 = Conv3D(f, (w, w, w), strides=1, padding='same', activation='relu',
                                  name='decoder_block4_conv1')(decoder_block4_features_forwarding)
    decoder_block4_add = add(
        [decoder_block3_upped, decoder_block4_conv1], name='decoder_block4_add')
    decoder_outputs = Conv3D(n_output_channels, (1, 1, 1), strides=1, padding='same', activation=final_activation_name,
                             kernel_initializer='zeros' if last_layer_zero_initialization else 'glorot_uniform',
                             activity_regularizer=last_activation_regularization,
                             name='decoder_block4_elementwise_filter')(decoder_block4_add)

    vnet_decoder = models.Model(inputs=[decoder_input_encoderblock1, decoder_input_encoderblock2,
                                        decoder_input_encoderblock3, decoder_input_encoderblock4,
                                        decoder_input_encoderblock5], outputs=decoder_outputs)
    vnet_decoder.name = 'vnet_decoder'

    if get_only_decoder:
        return None, None, vnet_decoder

    compressed_inputs_code = vnet_encoder(encoder_inputs)
    outputs = vnet_decoder(compressed_inputs_code)

    vnet = models.Model(inputs, outputs)
    try:
        plot_model(vnet, 'vnet_architecture.png', show_shapes=True)
        plot_model(vnet_encoder, 'vnet-encoder_architecture.png',
                   show_shapes=True)
        plot_model(vnet_decoder, 'vnet-decoder_architecture.png',
                   show_shapes=True)
    except OSError:
        pass

    # return vnet, vnet_encoder, vnet_decoder
    return vnet, vnet_encoder, vnet_decoder


def conc_input_handwritten_vnet(input_shape_per_image, final_activation_name, n_output_channels,
                                filter_width_normal_conv=5,
                                first_conv_n_filters=16, input_layer=None, activity_l2reg_coef=1e-3,
                                last_layer_zero_initialization=False, override_decoder_input_channels=None,
                                get_only_decoder=False, deform_regularisation=0):
    new_input_shape = input_shape_per_image[:-
                                            1] + (input_shape_per_image[-1] * 2,)
    registration_vnet, _, _ = handwritten_vnet(new_input_shape, 'sigmoid', 3,
                                               filter_width_normal_conv,
                                               first_conv_n_filters, input_layer, activity_l2reg_coef,
                                               last_layer_zero_initialization, override_decoder_input_channels,
                                               get_only_decoder, deform_regularisation)

    concatenated_source_target = Input(
        new_input_shape, name='concatenated_source_target')
    source = Input(input_shape_per_image, name='source_image')

    seed_registration_map = registration_vnet(concatenated_source_target)
    registration_map = Transformer.BuildRegistrationMap(
        name='registration_map_construction')(seed_registration_map)

    deformed_source = Transformer.diffeomorphicTransformer3D(name='registration_map_application')(
        [source, registration_map])

    model = models.Model(inputs=[concatenated_source_target, source],
                         outputs=[registration_map, deformed_source], name='concatenated_inputs_registration_model')
    return model


def get_registration_segmenter_vnet(input_shape, final_activation_name, n_output_channels, filter_width_normal_conv=5,
                                    first_conv_n_filters=16, with_target_segmenter=False, with_source_segmenter=False,
                                    registration_activity_l2reg_coef=1e-3,
                                    source_target_merge_operation='substraction', with_loss_trick=False,
                                    deform_regularisation=0):
    source = Input(input_shape, name='source_image')
    target = Input(input_shape, name='target_image')

    # Segmenter model: bicephalic input, shared encoder and decoder
    _, vnet_encoder, vnet_decoder_segmentation = handwritten_vnet(input_shape, final_activation_name,
                                                                  n_output_channels, filter_width_normal_conv,
                                                                  first_conv_n_filters)
    vnet_decoder_segmentation.name = 'decoder_segmentation'

    encoded_source = vnet_encoder(source)
    encoded_target = vnet_encoder(target)

    # Define decoders and models for segmentation
    segmenter_model_inputs = []
    segmenter_model_outputs = []
    if with_source_segmenter or with_target_segmenter:
        if with_source_segmenter:
            decoded_source = vnet_decoder_segmentation(encoded_source)
            decoded_source = Lambda(
                lambda x: x, name='decoder_segmentation_1')(decoded_source)
            segmenter_model_inputs.append(source)
            segmenter_model_outputs.append(decoded_source)
            # source_segmenter_model = models.Model(inputs=source, outputs=decoded_source)
        else:
            decoded_source = None
        if with_target_segmenter:
            decoded_target = vnet_decoder_segmentation(encoded_target)
            decoded_target = Lambda(
                lambda x: x, name='decoder_segmentation_2')(decoded_target)
            segmenter_model_inputs.append(target)
            segmenter_model_outputs.append(decoded_target)
            # target_segmenter_model = models.Model(inputs=target, outputs=decoded_target)
        else:
            decoded_target = None
        segmenter_model = models.Model(inputs=segmenter_model_inputs, outputs=segmenter_model_outputs,
                                       name='segmenter_model')
    else:
        decoded_source = None
        decoded_target = None
        segmenter_model = None

    # Merge source and target codes before registration decoder
    source_target_merge_operation = source_target_merge_operation.lower()
    merge_operations_layer = {'subtraction': subtract, 'addition': add,
                              'concatenation': concatenate}  # concatenate is axis -1 by default
    if source_target_merge_operation not in merge_operations_layer.keys():
        raise ValueError('Source-target merging operation {} not understood; supported operations: {}'.format(
            source_target_merge_operation, ['subtraction', 'addition', 'concatenation']))
    merge_operation_functional_layer = merge_operations_layer[source_target_merge_operation]

    encoded_source_minus_encoded_target = []
    for b, (source_block_code, target_block_code) in enumerate(zip(encoded_source, encoded_target)):
        encoded_source_minus_encoded_target.append(
            merge_operation_functional_layer([source_block_code, target_block_code],
                                             name='encoded_source_minus_encoded_target_%s_block%d' % (
                                                 source_target_merge_operation.lower(), b + 1))
        )

    if source_target_merge_operation == 'concatenation':
        override_decoder_input_channels = [code_s.get_shape().as_list()[-1] + code_t.get_shape().as_list()[-1]
                                           for code_s, code_t in zip(encoded_source, encoded_target)]
        print('override_decoder_input_channels',
              override_decoder_input_channels)
    # no need to change decoder input number of channels if merge operation keeps shapes (ie not concatenation)
    else:
        override_decoder_input_channels = None

    # Registration model: same input and encoder as segmenter model, outputs 3 channels mask with new decoder
    _, _, vnet_decoder_registration = handwritten_vnet(input_shape, 'sigmoid', 3, filter_width_normal_conv,
                                                       first_conv_n_filters,
                                                       activity_l2reg_coef=registration_activity_l2reg_coef,
                                                       last_layer_zero_initialization=False,
                                                       override_decoder_input_channels=override_decoder_input_channels,
                                                       get_only_decoder=True,
                                                       deform_regularisation=deform_regularisation)
    vnet_decoder_registration.name = 'vnet_decoder_registration'
    seed_registration_map = vnet_decoder_registration(
        encoded_source_minus_encoded_target)

    registration_map = Transformer.BuildRegistrationMap(
        name='registration_map_construction')(seed_registration_map)

    deformed_source = Transformer.diffeomorphicTransformer3D(name='registration_map_application')(
        [source, registration_map])

    # Loss trick: compute mask of non-tumour for both source and target, then multiply by the subtraction of
    # deformed source and target images, and add (afterwards) a MSE loss with 0: this is a trick equivalent
    # to the computation of masked MSE between deformed source and target with a mask of non-tumor regions
    if with_loss_trick:
        assert with_source_segmenter and with_target_segmenter, \
            'the loss trick necessitates the predictions of both tumor regions'

        predicted_background_mask_layer = Lambda(lambda t: K.cast(K.expand_dims(K.equal(K.argmax(K.stop_gradient(t),
                                                                                                 axis=-1),
                                                                                        0),
                                                                                -1),
                                                                  'float32'),
                                                 name='extract_predicted_mask')
        predicted_background_source_mask = predicted_background_mask_layer(
            decoded_source)
        predicted_background_target_mask = predicted_background_mask_layer(
            decoded_target)
        predicted_nontumor_mask = minimum([predicted_background_source_mask, predicted_background_target_mask],
                                          name='predicted_background_merged_mask')

        # Now compute the difference of the deformed source and the target, and multiply elementwise
        # by the non tumor regions
        deformed_source_minus_target = subtract([deformed_source, target],
                                                name='deformed_source_minus_target')
        predicted_tumor_masked_deformed_source_minus_target = multiply([deformed_source_minus_target,
                                                                        predicted_nontumor_mask],
                                                                       name='registration_map_application_losstrick')

        registration_model = models.Model(inputs=[source, target],
                                          outputs=[predicted_tumor_masked_deformed_source_minus_target,
                                                   predicted_nontumor_mask],
                                          name='registration_model')  # needs to be trained using MSE(0, y_pred)

    else:
        registration_model = models.Model(
            inputs=[source, target], outputs=deformed_source, name='registration_model')

    outputs = []
    if with_source_segmenter:
        outputs.append(decoded_source)
    outputs.append(registration_map)
    outputs.append(deformed_source)
    if with_loss_trick:
        outputs.extend(
            [predicted_tumor_masked_deformed_source_minus_target, predicted_nontumor_mask])
    if with_target_segmenter:
        outputs.append(decoded_target)
    multitask_model = models.Model(
        inputs=[source, target], outputs=outputs, name='multitask_model')

    # Plots models
    try:
        if with_target_segmenter or with_source_segmenter:
            plot_model(segmenter_model, 'segmenter_model.png',
                       show_shapes=True)

        plot_model(registration_model, 'registration_model.png',
                   show_shapes=True, rankdir='LR')
        plot_model(multitask_model, 'multitask_model.png',
                   show_shapes=True, rankdir='LR')
    except OSError:
        pass
    return multitask_model, segmenter_model, registration_model
