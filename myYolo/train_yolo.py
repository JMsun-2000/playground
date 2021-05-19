#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:23:35 2021

@author: sunjim
"""

import numpy as np
import os
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from my_yolo import (YoloBody, yolo_loss)
import tensorflow as tf


# Default anchor boxes
# assign an object to an anchors, which is highest IoU
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main():
    anchors = YOLO_ANCHORS
    class_names = get_classes('model_data/pascal_classes.txt')
    print(class_names)
    
    data_path = ""
    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'
    image_data, boxes = process_data(data['images'], data['boxes'])
    
    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)
    
    model_body, model = create_model(anchors, class_names)
    
    train(
        model,
        class_names,
        anchors,
        image_data,
        boxes,
        detectors_mask,
        matching_true_boxes
    )
  
    
def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    # divides up the image into a grid of 13*13 cells
    # each of these celss is responsible for predicting 5 bounding boxes
    detectors_mask_shape = (13, 13, 5, 1) 
    # a bounding box describles a rectangle that encloses an object + a confidence score if it's an object
    # x, y, w, h, c
    matching_boxes_shape = (13, 13, 5, 5)
    
    # model input layers
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)
    
    # create model body
    yolo_model = YoloBody(image_input, len(anchors), len(class_names))
    yolo_model.summary()
    # original darknet layers, model.layers[-1] return last layer
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    topless_yolo.summary()
    
    if load_pretrained:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        """
        if not os.path.exists(topless_yolo_path):
            print("create topless weights file")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model('model_data/yolo.h5')
            # ignore layer 20, 21 added by me, use original darknet layers
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        """
        topless_yolo.load_weights(topless_yolo_path)
        # debug
        # for layer in topless_yolo.layers:
        #     weights= layer.get_weights()
        #     print(weights)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    # last layer is replaced
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)
    model_body = Model(image_input, final_layer)
    
    #debug
    # from keras import backend as K
    # yolo_output = model_body.output
    # print(yolo_output.shape)
    # yolo_output_shape = K.shape(yolo_output)
    # print(yolo_output_shape.shape)
    # feats = K.reshape(model_body.output, [
    #     -1, yolo_output_shape[1], yolo_output_shape[2], len(anchors),
    #     len(class_names) + 5
    # ])
    # print(feats)
    # print(feats[..., 0:2])
    # print(feats[..., 2:4])
    #debug end
    
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        model_loss = Lambda(yolo_loss, output_shape=(1, ), name='yolo_loss', 
                            arguments={'anchors': anchors,
                                        'num_classes': len(class_names)
                                })([model_body.output, boxes_input, 
                                    detectors_mask_input, matching_boxes_input])
    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)
    
    return model_body, model

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    # for the last layer has outputed the loss, so y_pred is loss. 
    # lambda y_true, y_pred: y_pred
    # func(y_true, y_pred):
    #   return y_pred
    # end
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred
            }) # This is a hack to use the custom loss function in the last layer.
    
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])
    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')    

def get_classes(classes_path):
    with open(classes_path) as f:
        # get an array, every element is a line
        class_names = f.readlines()
        #print(class_names)
    # strip() could remove blank or enter at the end of string
    class_names = [c.strip() for c in class_names]
    return class_names        