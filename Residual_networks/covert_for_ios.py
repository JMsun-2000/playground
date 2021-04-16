#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:01:39 2020

@author: sunjim
"""

from keras.models import Model, load_model, model_from_json
import coremltools


#import tensorflow as tf


# from resnets_utils import *
# import coremltools.proto.FeatureTypes_pb2 as ft 
# from keras.preprocessing import image
# import numpy as np
# from keras import layers
# from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from keras.models import Model, load_model
# from keras.preprocessing import image
# from keras.utils import layer_utilxs
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
# from resnets_utils import *
# from keras.initializers import glorot_uniform
# import scipy.misc
# from matplotlib.pyplot import imshow
# import PIL
# numpy.set_printoptions(threshold=sys.maxsize)

# import keras.backend as K
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)

# load json and create model
json_file = open('generated_models/trainedModelstruct.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("generated_models/trainedModelweight.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load test data
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# #print(classes)
# #X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_pokemon_dataset()
# # print(Y_test_orig)
# # print(Y_train_orig)
# # Normalize image vectors
# X_train = X_train_orig/255.
# X_test = X_test_orig/255.
# # X_train = X_train_orig
# # X_test = X_test_orig
# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T

# preds = loaded_model.evaluate(X_test, Y_test)
# print ("Loss = " + str(preds[0]))
# print ("Test Accuracy = " + str(preds[1]))

loaded_model.save('try.h5')

# img_path = 'images/double_check_lapras.jpg'
# # imgopen = PIL.Image.open(img_path)
# # imgopen.show()

# img = image.load_img(img_path, target_size=(64, 64))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# x = x/255.0
# # print(x)
# print('Input image shape:', x.shape)
# print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
# print(loaded_model.predict(x))

# convert
output_labels = ['0', '1', '2', '3', '4', '5']
# your_model=coremltools.convert(loaded_model, source="tensorflow")
your_model = coremltools.convert('try.h5', source='tensorflow')
# your_model = coremltools.converters.keras.convert('myWholeModel.h5')
#your_model = coremltools.converters.keras.convert('try.h5', input_names=['image'], output_names=['output'], class_labels=output_labels, image_input_names='image')

# your_model.author = 'Jiaming Sun'
# your_model.short_description = 'Digit Recognition with MNIST'
# your_model.input_description['image'] = 'Takes as input an image'
# your_model.output_description['output'] = 'Prediction of Digit'

your_model.save('try.mlmodel')

# spec = coremltools.utils.load_spec("try.mlmodel")

# input = spec.description.input[0]
# input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
# input.type.imageType.height = 64 
# input.type.imageType.width = 64

# coremltools.utils.save_spec(spec, "YourNewModel.mlmodel")