#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:49:26 2021

@author: sunjim
"""

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from my_utils.utils import compose
import functools
from functools import partial
from keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization

# Partial wrapper for Convolution2D with static default argument.
_DarknetConv2D = partial(Conv2D, padding='same')

@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs.update(kwargs)
    return _DarknetConv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
        )

def bottleneck_block(outer_filters, bottleneck_filters):
    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3))
        )

def bottleneck_x2_block(outer_filters, bottleneck_filters):
    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3))
        )

def darknet_body():
    """Generate first 18 conv layers of Darknet-19."""
    return compose(
        DarknetConv2D_BN_Leaky(32, (3,3)), #layer1
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)), #2
        MaxPooling2D(),
        bottleneck_block(128, 64), #4~6
        MaxPooling2D(),
        bottleneck_block(256, 128), #7~9
        MaxPooling2D(),
        bottleneck_x2_block(512, 256), #10~14
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512) #15~19
        )