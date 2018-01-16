# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------
#   Modified U-Net models for the Carvana Image Masking challenge
#   Using Keras with the TensorFlow backend
#   There are two models:
#   1) The dilated_unet model uses dilated convolutions
#   2) the resblock_unet model uses residual block
#
#   Meena Mani
#   December 2017
#--------------------------------------------------------------------------------


from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K


from losses import (
	binary_crossentropy, 
	dice_loss, 
	bce_dice_loss, 
	dice_coef, 
	weighted_bce_dice_loss
)

#--------------------------------------------------------------------------------
# Model 1: Modified UNet with dilation blocks
# Author lyakaap
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/model.py 
#--------------------------------------------------------------------------------

def get_dilated_unet(input_shape=(512, 512, 3), init_nb=44, lr=0.0001, loss=bce_dice_loss):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
    down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked dilated convolution
    dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=1)(down3pool)
    dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=4)(dilate2)
    dilate4 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=8)(dilate3)
    dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=16)(dilate4)
    dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', 
		     padding='same', dilation_rate=32)(dilate5)
    dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
    
    up3 = UpSampling2D((2, 2))(dilate_all_added)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model

#--------------------------------------------------------------------------------
# Model 2: Modified UNet with residual blocks
# Author Meena Mani
# The encoder and decoder paths have been taken from @lyakaap
# The identity block function have been adapted from ResNet-50
#--------------------------------------------------------------------------------

# Resnet50  Identity block
# from https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py

def identity_block(input_tensor, kernel_size, filters, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        block: '1','2'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'bottle_' + 'res_' + str(block) 
    bn_name_base = 'bottle_' + 'bn_' + str(block)
    act_name_base = 'bottle_' + 'act_' + str(block)

    x = Conv2D(filters1, (1, 1), name=conv_name_base + 'a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = Activation('relu', name=act_name_base + 'a')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + 'b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = Activation('relu', name=act_name_base + 'b')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + 'c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)

    x = add([x, input_tensor])
    x = Activation('relu', name=act_name_base + 'c')(x)
    return x


# encoder function
def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu'):
    
    skip = []
    for i in range(n_block):
        conv_name_base = 'enc_' + 'conv_' + str(i + 1) 
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, 
                   padding='same', name=conv_name_base + 'a')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation,
                   padding='same', name=conv_name_base + 'b')(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip

# bottleneck region
def bottleneck(x, filters, depth=5,
               kernel_size=(3, 3), activation='relu'):

    residual_blocks = []
    x = Conv2D(filters*8, (3, 3), activation='relu', 
               padding='same', name='bottle_conv_1')(x)
    for i in range(depth):
        x = identity_block(x, 3, [filters*2, filters*2, filters*8], block=i+2)
        residual_blocks.append(x)
     return add(residual_blocks)

# decoder function
def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu'):

    for i in reversed(range(n_block)):
        conv_name_base = 'dec_' + 'conv_' + str(i + 1) 
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation, 
                   padding='same', name=conv_name_base + 'a')(x)
        x = concatenate([skip[i], x])
        x = Conv2D(filters * 2**i, kernel_size, activation=activation,
                   padding='same', name=conv_name_base + 'b')(x)
        x = Conv2D(filters * 2**i, kernel_size, activation=activation,
		   padding='same', name=conv_name_base + 'c')(x)
    return x


def get_resblock_unet(
        input_shape=(512, 512, 3),
        filters=44,
        n_block=3,
        lr=0.0001,
        loss=bce_dice_loss,
        n_class=1
):
    inputs = Input(input_shape)
    
    enc, skip = encoder(inputs, filters, n_block)
    bottle = bottleneck(enc, filters=filters, depth=5)
    dec = decoder(bottle, skip, filters, n_block)
    classify = Conv2D(n_class, (1, 1), activation='sigmoid')(dec)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

return model

############################


""" Shown below is the full expanded form of the resblock_unet function

def get_resblock_unet(input_shape=(512, 512, 3), filters=44, lr=0.0001, loss=bce_dice_loss):
    
    inputs = Input(input_shape)
    
    down1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    down1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(down1)
    down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    
    down2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(down1pool)
    down2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(down2)
    down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
    down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
    down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    
    # stacked residual blocks
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', 
                   padding='same', name='bottle_conv_1')(down3pool)
    res1 = identity_block(conv4, 3, [filters*2, filters*2, filters*8], block='2')
    res2 = identity_block(res1,  3, [filters*2, filters*2, filters*8], block='3')
    res3 = identity_block(res2,  3, [filters*2, filters*2, filters*8], block='4')
    res4 = identity_block(res3,  3, [filters*2, filters*2, filters*8], block='5')
    res5 = identity_block(res4,  3, [filters*2, filters*2, filters*8], block='6')
    res_all_added = add([conv4, res1, res2, res3, res4, res5])
    
    up3 = UpSampling2D((2, 2))(res_all_added)
    up3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = concatenate([down3, up3])
    up3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(up3)

    up2 = UpSampling2D((2, 2))(up3)
    up2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = concatenate([down2, up2])
    up2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(up2)
    
    up1 = UpSampling2D((2, 2))(up2)
    up1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(up1)
    up1 = concatenate([down1, up1])
    up1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(up1)
    
    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr), loss=loss, metrics=[dice_coef])

    return model

"""
