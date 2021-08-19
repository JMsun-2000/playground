#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:59:52 2021

@author: sunjim
"""


from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Lambda, Conv2D
from tensorflow.keras import backend as K

from my_utils.utils import compose
from my_utils.keras_darknet19 import (DarknetConv2D, DarknetConv2D_BN_Leaky,
                              darknet_body)

import tensorflow as tf
import numpy as np

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

CLASS_NUM = 20


# def MyYolo(input_shape = (608, 608, 3)):
#     X_input = Input(input_shape)
#     # Stage 1
#     X = Conv2D(32, (1, 1), strides = (1, 1), name = 'conv2d_1', kernel_initializer = glorot_uniform(seed=0))(X)
    
def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
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
    
def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    #anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    anchors_tensor = K.reshape(anchors, [1, 1, 1, num_anchors, 2])
    
    # Static implementation for fixed models.
    # TODO: Remove or add option for static implementation.
    # _, conv_height, conv_width, _ = K.int_shape(feats)
    # conv_dims = K.variable([conv_width, conv_height])
    
    # Dynamic implementation of conv dims for fully convolutional model.
    # shape(feats) (m, gridx, gridy, anchors, xxx)
    # get grids shape (13, 13)
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    #conv_dims = [13, 13]
    # conv_dims = feats.shape[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    # will generate array [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    conv_height_index = K.arange(0, stop=conv_dims[0])
    """
    quick index from grid number to height index
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  0,  1,  2,  3,
        4,  5,  6,  7,  8,  9, 10, 11, 12, 
        ....
       10, 11, 12,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
      dtype=int32)>
    """
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    """
    quick index from grid number to width index, for there is no direct function
    So add a dimension at index 0 to make a 2d matrix
    Then transpose it and flatten it again, get
    
    array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        ....
        9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
      dtype=int32)>

    """
    conv_width_index = K.arange(0, stop=conv_dims[1])
    #  array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # ....
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]], dtype=int32)>
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    """
    array([[ 0,  0],
       [ 1,  0],
       [ 2,  0],
       [ 3,  0],
       ...
       [ 9, 12],
       [10, 12],
       [11, 12],
       [12, 12]], dtype=int32)>
    """
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # shape (169, 2) to (1, 13, 13, 1, 2)
    # w,h index with same dimenation as feats.
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    # cast it as features dtype, here is from int32 to float32
    conv_index = K.cast(conv_index, K.dtype(feats))
    # reshape features by adding anchors axis
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    # <tf.Tensor: shape=(1, 1, 1, 1, 2), dtype=int32, numpy=array([[[[[13, 13]]]]], dtype=float32)>
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    
    # get all sigmoid x, y
    box_xy = K.sigmoid(feats[..., :2])
    # exp w, h
    box_wh = K.exp(feats[..., 2:4])
    # sigmoid pc
    box_confidence = K.sigmoid(feats[..., 4:5])
    # softmax classes
    box_class_probs = K.softmax(feats[..., 5:])
    
    # Adjust preditions to each spatial grid point and anchor size.
    # Note: YOLO iterates over height index before width index.
    # xy from 1 grid to whole image
    box_xy = (box_xy + conv_index) / conv_dims
    # wh under 5 different anchors' tensor
    #box_wh = box_wh * anchors_tensor / conv_dims
    box_wh = box_wh * K.cast(anchors_tensor, K.dtype(box_wh)) / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_box_translate(yolo_output):
    yolo_output_shape = K.shape(yolo_output)
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(yolo_output, YOLO_ANCHORS, CLASS_NUM)
    boxes = yolo_boxes_to_corners(pred_xy, pred_wh)
    box_scores = pred_confidence * pred_class_prob

    print(boxes.shape)
    print(pred_confidence.shape)
    print(pred_class_prob.shape)
    
    box_class_scores = K.max(box_scores, axis=-1, keepdims=True)
    box_classes = K.cast(K.expand_dims(K.argmax(box_scores, axis=-1)), K.dtype(box_class_scores))
    print("--------output---------")
    print(box_class_scores.shape)
    print(box_classes.shape)
    print(K.concatenate([boxes, box_class_scores, box_classes]).shape)
    return K.concatenate([boxes, box_class_scores, box_classes])

def yolo_loss(args, anchors, num_classes, rescore_confidence=False, print_loss=False):
    (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
    num_anchors = len(anchors)
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1
    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(yolo_output, anchors, num_classes)
    
    # Unadjusted box predictions for loss.
    # TODO: Remove extra computation shared with yolo_head.
    yolo_output_shape = K.shape(yolo_output)
    # yolo_output (None, 13, 13, anchor*(5+classes))
    feats = K.reshape(yolo_output, [-1, yolo_output_shape[1], yolo_output_shape[2], num_anchors, num_classes + 5])
    # for one anchor inner index
    # x, y: 0, 1
    # w, h: 2, 3
    # confidence: 4
    # classes: 5~end
    pred_boxes = K.concatenate(
        (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)
    
    # TODO: Adjust predictions by image width/height for non-square images?
    # IOUs may be off due to different aspect ratio.

    # Expand pred x,y,w,h to allow comparison with ground truth.
    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    
    # Input(shape=(None, 5)) ~>  <tf.Tensor 'input_16:0' shape=(None, None, 5) dtype=float32>
    true_boxes_shape = K.shape(true_boxes)
     # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = K.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    # max mins(top_left) is top_left of intersect part
    intersect_mins = K.maximum(pred_mins, true_mins)
    # min max(bottom_right) is bottom_right of intersect part
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    # The perfect situation is interset_areas=union_areas
    iou_scores = intersect_areas / union_areas
    
    # Best IOUs for each location.
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious)
    
    # A detector has found an object if IOU > thresh for some true box.
    # thresh bool to float
    object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))
    
    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.
    
    
    """
    # divides up the image into a grid of 13*13 cells
    # each of these celss is responsible for predicting 5 bounding boxes
    detectors_mask_shape = (13, 13, 5, 1) 
    detectors_mask = Input(shape=detectors_mask_shape)
    object_scale = 5
    no_object_scale = 1
    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    
    object_detections =1  => no_object_weights=0 since no_object_weight has been 0, there is no lost no_objects_loss=0
    if no_object, just care pc loss, and pc = 0 is the best
    
    object_detections =0 => no_ojbect_weights= no_object_scale*(1 - detectors_mask) # 1-detectors_mask=no_detectors_mask
    no_objects_loss is 0 - pc. For 0 is no object, 1 is object detected.
    
    """
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * K.square(0 - pred_confidence)
    
    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        K.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        K.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss
    
    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
    matching_classes = K.one_hot(matching_classes, num_classes)
    classification_loss = (class_scale * detectors_mask *
                           K.square(matching_classes - pred_class_prob))
    
    # Coordinate loss for matching detection boxes.
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        K.square(matching_boxes - pred_boxes))
    
    # Sum all loss
    confidence_loss_sum = K.sum(confidence_loss)
    classification_loss_sum = K.sum(classification_loss)
    coordinates_loss_sum = K.sum(coordinates_loss)
    total_loss = 0.5 * (
        confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)
    if print_loss:
        total_loss = tf.print(
            total_loss, [
                total_loss, confidence_loss_sum, classification_loss_sum,
                coordinates_loss_sum
            ],
            message='yolo_loss, conf_loss, class_loss, box_coord_loss:')

    return total_loss

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filter YOLO boxes based on object and class confidence."""
    box_scores = box_confidence * box_class_probs
    box_class_scores = K.max(box_scores, axis=-1)
    prediction_mask = box_class_scores >= threshold
    
    # TODO: Expose tf.boolean_mask to Keras backend?
    box_classes = K.argmax(box_scores, axis=-1)
    boxes = tf.boolean_mask(boxes, prediction_mask)
    scores = tf.boolean_mask(box_class_scores, prediction_mask)
    classes = tf.boolean_mask(box_classes, prediction_mask)
    return boxes, scores, classes

def yolo_eval_with_sess(yolo_outputs,
              image_shape,
              sess,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)
    
   # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
   

    # TODO: Something must be done about this ugly hack!
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    sess.run(tf.compat.v1.variables_initializer([max_boxes_tensor]))
    #Greedily selects a subset of bounding boxes in descending order of score
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes
    
def yolo_eval(yolo_outputs,
              image_shape,
              max_boxes=10,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input batch and return filtered boxes."""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    boxes, scores, classes = yolo_filter_boxes(
        boxes, box_confidence, box_class_probs, threshold=score_threshold)
    
   # Scale boxes back to original image shape.
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
   

    # TODO: Something must be done about this ugly hack!
    with tf.compat.v1.Session() as sess:
        max_boxes_tensor = K.variable(max_boxes, dtype='int32')
        sess.run(tf.compat.v1.variables_initializer([max_boxes_tensor]))
    sess.close()
    #Greedily selects a subset of bounding boxes in descending order of score
    nms_index = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    scores = K.gather(scores, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, scores, classes

def preprocess_true_boxes(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array of boxes in 1 sample image
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    # true_boxes is [m, box_cnt, 5], 5 is mid_x, mid_y, width, height, class
    # image_size would be 416*416
    # going to find best IOU to pair anchor box with true_boxes
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    # 416/32 = 13 grids
    conv_height = height // 32
    conv_width = width // 32
    num_box_params = true_boxes.shape[1] # should be 5
    detectors_mask = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)
    
    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        # [x, y, w, h] * [13, 13, 13, 13]
        box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
        # round down, 0.0~1.0 to 0~12 grid index
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes
            '''
            (-a, -b)
              ┏━━━━━━━━━━━━━━━━┓
              ┃                ┃
              ┃  (-x, -y)      ┃
              ┃      ┏━━━━━━━━━╋━━━━━━━┓
              ┃      ┃  (a, b) ┃       ┃
              ┗━━━━━━╋━━━━━━━━━┛       ┃
                     ┗━━━━━━━━━━━━━━━━━┛(x, y)
            '''
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k
            
        if best_iou > 0:
            # set box to this anchor
            detectors_mask[i, j, best_anchor] = 1
            # convert box x, y coordinates to relative one in its grid
            # convert box w, h relative to ist anchor's size
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
                    np.log(box[2] / anchors[best_anchor][0]), np.log(box[3] / anchors[best_anchor][1]),
                    box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    # all set
    return detectors_mask, matching_true_boxes
                

def test_create_model():
    image_input = Input(shape=(608, 608, 3))
    yolo_model = YoloBody(image_input, 5, 5)
    yolo_model.summary()


# test_create_model()











