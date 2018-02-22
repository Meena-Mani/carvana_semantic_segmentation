# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------
#   LinkNet in Keras for the Carvana Image Masking challenge
#   Some of the helper functions have been adapted from ResNet-18
#   https://github.com/raghakot/keras-resnet/blob/master/resnet.py
#
#   Author: Meena Mani
#   January 19 2017
#--------------------------------------------------------------------------------

from __future__ import division

from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D 
from keras.layers import BatchNormalization, Activation, Reshape, Permute
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
import keras.backend as K

from losses import (
	binary_crossentropy, 
	dice_loss, 
	bce_dice_loss, 
	dice_coef, 
	weighted_bce_dice_loss
)


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _conv_bn(**conv_params):
    """Helper to build a conv -> BN block 
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    name = conv_params["name"]

    def f(input):
        x =  Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=name + '_conv')(input)
        x = BatchNormalization(axis=CHANNEL_AXIS, name=name + '_bn')(x)
        return x 
    return f


def _shortcut(input, residual, name):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
		          name = name)(input)

    return add([shortcut, residual])


def encoder_block(input, kernel_size, filters, block):
    """A 4 layer block that has a shortcut after every two convolutions
    # Arguments
        input: input tensor
        kernel_size: default 3 
        filters: list of integers, the input and output filters
        block: '1','2'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that the first conv layer has strides=(2,2)
    """

    conv_bn_name_base = 'encoder_' + str(block) + '_'

    i_filters, o_filters = filters

    x1 = _conv_bn(filters=o_filters, kernel_size=(3,3), strides=(2,2),
                 name=conv_bn_name_base + '1a')(input)
    x1 = Activation('relu', name= conv_bn_name_base + '1a_act')(x1)

    x1 = _conv_bn(filters=o_filters, kernel_size=(3,3),
                 name=conv_bn_name_base + '1b')(x1)
    x1 = _shortcut(input, x1, name= conv_bn_name_base + '1b_shortcut')
    x1 = Activation('relu', name= conv_bn_name_base + '1b_act')(x1)

    x2 = _conv_bn(filters=o_filters, kernel_size=(3,3), 
                 name=conv_bn_name_base + '2a')(x1)
    x2 = Activation('relu', name= conv_bn_name_base + '2a_act')(x2)

    x2 = _conv_bn(filters=o_filters, kernel_size=(3,3), 
                 name=conv_bn_name_base + '2b')(x2)
    x2 = _shortcut(x1, x2, name=conv_bn_name_base + '2b_shortcut')
    x2 = Activation('relu', name= conv_bn_name_base + '2b_act')(x2)

    return x2


def decoder_block(input, filters, block):
    """The decoder block is the upsampling block 
    # Arguments
        input: input tensor
        filters: list of integers, the number of input and output filters 
        block: '1','2'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    conv_bn_name_base = 'decoder_' + str(block) + '_'

    i_filters, o_filters = filters

    x = _conv_bn(filters=i_filters//4, kernel_size=(1, 1), 
                 name=conv_bn_name_base + '1a')(input)
    x = Activation('relu', name= conv_bn_name_base + '1a_act')(x)

    x = Conv2DTranspose(filters=i_filters//4, kernel_size=(3,3),
               strides=(2,2), padding='same', name=conv_bn_name_base +'1b_fullconv')(x)
    x = BatchNormalization(axis=CHANNEL_AXIS, name=conv_bn_name_base + '1b_bn')(x)
    x = Activation('relu', name= conv_bn_name_base + '1b_act')(x)

    x = _conv_bn(filters=o_filters, kernel_size=(1, 1), 
                 name=conv_bn_name_base + '1c')(x)
    x = Activation('relu', name= conv_bn_name_base + '1c_act')(x)

    return x


##------------------------------------------------------

def build_LinkNet(
	input_shape=(512, 512, 3), 
	nb_filters=64, 
	n_classes=1, 
	lr=0.0001, 
	loss=bce_dice_loss
):


    _handle_dim_ordering()
    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")


    input = Input(shape=input_shape)

    ## -------   initial block
    conv1 = _conv_bn(filters=nb_filters, kernel_size=(7, 7), strides=(2, 2), 
            name='initial_')(input)
    conv1 = Activation('relu', name='initial_act')(conv1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1) 
    

    ## -------   encoder
    down1 = encoder_block(pool1, (3,3), [nb_filters,  nb_filters],   '1')
    down2 = encoder_block(down1, (3,3), [nb_filters,  nb_filters*2], '2')
    down3 = encoder_block(down2, (3,3), [nb_filters*2,nb_filters*4], '3')
    down4 = encoder_block(down3, (3,3), [nb_filters*4,nb_filters*8], '4')
    

    ## -------  decoder 
    up4 = decoder_block(down4, [nb_filters*8,nb_filters*4], '4')

    up3 = concatenate([up4, down3])
    up3 = decoder_block(up3, [nb_filters*4,nb_filters*2], '3')

    up2 = concatenate([up3, down2])
    up2 = decoder_block(up2, [nb_filters*2,nb_filters],'2')

    up1 = concatenate([up2, down1])
    up1 = decoder_block(up1, [nb_filters,nb_filters], '1')


    ## -------   final conv block
    name_base = 'final' +  '_'

    final = Conv2DTranspose(nb_filters//2, kernel_size=(3,3), strides=(2,2), 
                        padding='same', name=name_base +'1_fullconv')(up1)
    final = BatchNormalization(axis=CHANNEL_AXIS, name=name_base + '1_bn')(final)
    final = Activation('relu', name=name_base + '_1_act')(final)

    final = _conv_bn(filters=nb_filters//2, kernel_size=(3, 3), 
                     name=name_base + '2')(final)
    final = Activation('relu', name=name_base +'_2_act')(final)

    final = Conv2DTranspose(n_classes, kernel_size=(2,2), strides=(2,2),
                       padding='same', name=name_base + '3_fullconv')(final)


    ## -------  classifier block 
    classify = Activation('sigmoid')(final)

    model = Model(inputs=input, outputs=classify)

    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model

