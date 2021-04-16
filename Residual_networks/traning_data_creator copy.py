#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:37:21 2020

@author: sunjim
"""

import h5py
import os
import numpy as np
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join

# img_path = 'images/my_two_square.jpg'
# img = image.load_img(img_path, target_size=(64, 64))
# x = image.img_to_array(img)
# print('Input image shape:', x.shape)
# print('Input image type:', x.dtype)
# f = h5py.File('my_datasets/test.h5', 'w')
# dset = f.create_dataset('test', shape=(2, 64, 64, 3), maxshape=(None, 64, 64, 3))
# dset[0]=x
# dset[1]=x
# print(dset)
# f.close()

# test
 test_dataset = h5py.File('datasets/test_signs.h5', "r")
 test_set_x_orig = test_dataset["test_set_x"][:] # your test set features
 test_set_y_orig = np.array(test_dataset["test_set_y"][:])
 # print(test_set_x_orig)
 print(test_set_x_orig.shape)
 num_data = len(test_set_y_orig)
 print(num_data)
 perm = np.random.permutation(num_data)
 print(perm)
 print(test_set_x_orig[perm])
 

img_folder = 'my_datasets/test_train'
f = h5py.File('my_datasets/test.h5', 'w')
train_size = 5
train = []
# will be 1200
dset = f.create_dataset('train_set_x', shape=(train_size, 64, 64, 3))
lset = f.create_dataset('train_set_y', shape=(train_size, 1))
file_count = 0
for filename in listdir(img_folder):
    cur_file = join(img_folder, filename)
    print(cur_file)
    img = image.load_img(cur_file, target_size=(64, 64))
    x = image.img_to_array(img)
    train.append(x)
    dset[file_count] = x
    lset[file_count] = file_count
    file_count += 1
print(dset[:, 32, 32, :])
print(lset[:])
print(np.array(train,dtype='float32').shape)
# work = np.array(dset)
#shuffle
perm = np.random.permutation(train_size)
dset = np.array(dset)[perm]
lset = np.array(lset)[perm]
print(dset[:, 32, 32, :])
print(lset[:])
f.close()

double_check = h5py.File('my_datasets/test.h5', 'r')
check_x =  np.array(double_check["train_set_x"][:]) # your test set features
check_y =  np.array(double_check["train_set_y"][:])
print(check_x[:, 32, 32, :])
print(check_y[:])
double_check.close()




