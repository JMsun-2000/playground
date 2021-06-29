#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:22:51 2021

@author: sunjim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:23:35 2021

@author: sunjim
"""

import numpy as np
import io
import os
import PIL
import h5py
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from my_yolo import (YoloBody, yolo_loss, preprocess_true_boxes, yolo_head, yolo_eval)
from my_utils.draw_boxes import draw_boxes
import tensorflow as tf
from keras import backend as K


# Default anchor boxes
# assign an object to an anchors, which is highest IoU
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    anchors = YOLO_ANCHORS
    class_names = get_classes('model_data/pascal_classes.txt')
    print(class_names)
    
    data_path = "train_data/pascal_voc_07_12.hdf5"
    voc = h5py.File(data_path, 'r')
    '''
    # real train data
    train_image = np.array(voc['train/images'])
    train_box = np.array(voc['train/boxes'])
    '''
    
    # test happy path logic, just use 2 samples
    train_image = np.array([voc['train/images'][28], voc['train/images'][29]])
    train_box = np.array([voc['train/boxes'][28], voc['train/boxes'][29]])
    #data = np.array(voc["train"][:]) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'
    image_data, boxes_data = process_data(train_image, train_box)
    
    detectors_mask, matching_true_boxes = get_detector_mask(boxes_data, anchors)
    
    model_body, model = create_model(anchors, class_names)
    
    train(
        model,
        class_names,
        anchors,
        image_data,
        boxes_data,
        detectors_mask,
        matching_true_boxes
    )
    
    draw(model_body,
        class_names,
        anchors,
        image_data,
        image_set='val', # assumes training/validation split is 0.9
        weights_name='trained_stage_3_best.h5',
        save_all=False
        )
  
def process_data(images, boxes=None):
    images = [PIL.Image.open(io.BytesIO(i)) for i in images]
    orig_sizes = np.array([np.array([image.width, image.height]) for image in images])
    #orig_size = np.array([images[0].width, images[0].height])
    #orig_size2 = np.expand_dims(orig_size, axis=0)
    
     # Image preprocessing. uniform to size 416 * 416, RGB 0~255 to 0~1
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]
    
    if boxes is not None:
        # boxes[m, box_cnt, 5]
        # Box preprocessing.
        # Original boxes stored as 1D list of [class, x_min, y_min, x_max, y_max].
        boxes = [box.reshape((-1, 5)) for box in boxes] 
        # Get extents as [y_min, x_min, y_max, x_max, class] for comparision with
        # model output. not use
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters from [class, x_min, y_min, x_max, y_max] to [x_center, y_center, box_width, box_height, class].
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3])/orig_sizes[idx] for idx, box in enumerate(boxes)]
        boxes_wh = [(box[:, 3:5] - box[:, 1:3])/orig_sizes[idx] for idx, box in enumerate(boxes)]
        # boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        # boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)] 
        
        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
                
        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))
                
        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)
    
def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    # initial 0 for every sample (m)
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        print(i)
        print(box)
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])
        
    return np.array(detectors_mask), np.array(matching_true_boxes)
    
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
    
def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        # get 0~90% image as train
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data[:int(len(image_data)*.9)]]) 
    elif image_set == 'val':
        # last 10% as valid
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data[int(len(image_data)*.9):]])    
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
        
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)
    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # session.run(any function about tf.placeholder, feed_dict={placeholder_name: input_value, ...})
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)
        
        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()

def get_classes(classes_path):
    with open(classes_path) as f:
        # get an array, every element is a line
        class_names = f.readlines()
        #print(class_names)
    # strip() could remove blank or enter at the end of string
    class_names = [c.strip() for c in class_names]
    return class_names        