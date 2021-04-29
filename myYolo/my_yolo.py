#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:59:52 2021

@author: sunjim
"""


from keras.models import Model
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.layers import Input, Lambda, Conv2D

from my_utils.utils import compose
from my_utils.keras_darknet19 import (DarknetConv2D, DarknetConv2D_BN_Leaky,
                              darknet_body)

import tensorflow as tf


# def MyYolo(input_shape = (608, 608, 3)):
#     X_input = Input(input_shape)
#     # Stage 1
#     X = Conv2D(32, (1, 1), strides = (1, 1), name = 'conv2d_1', kernel_initializer = glorot_uniform(seed=0))(X)
    
def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    return tf.nn.space_to_depth(x, block_size=2)    

def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def YoloBody(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body()(inputs))
    conv20 = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(1024, (3, 3))
        )(darknet.output)
    # layer No.43 is the last layer of conv13
    conv13 = darknet.layers[43].output
    conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13)
    conv21_reshaped = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='Lambda_space_depth'
        )(conv21)
    x = concatenate([conv21_reshaped, conv20])
    x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
    # every anchor should have classes, pc, x, y, w, h
    x = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x) 
    return Model(inputs, x)

def yolo(inputs, anchors, num_classes):
    num_anchors = len(anchors)
    body = YoloBody(inputs, num_anchors, num_classes)
    

def test_create_model():
    image_input = Input(shape=(608, 608, 3))
    yolo_model = YoloBody(image_input, 5, 5)
    yolo_model.summary()


test_create_model()
