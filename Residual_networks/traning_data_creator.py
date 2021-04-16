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
 
word_dict = dict()
word_dict['bulbasaur'] = 0
word_dict['charmander'] = 1
word_dict['squirtle'] = 2
word_dict['pikachu'] = 3
word_dict['lapras'] = 4
word_dict['mewtwo'] = 5

def create_image_array_from_a_folder(img_folder, label_value):
    train_sets = []
    train_labels = []
    for filename in listdir(img_folder):
        if filename.startswith('.'):
            continue
        cur_file = join(img_folder, filename)
        print(cur_file)
        img = image.load_img(cur_file, target_size=(64, 64))
        x = image.img_to_array(img)
        train_sets.append(x)
        train_labels.append(label_value)
    train_count = len(train_labels)   
    return train_count, np.array(train_sets, dtype='float'), np.array(train_labels)
    

def create_dataset_from_folder(top_folder, target_file, data_set_name, data_lable_name):
    all_set = []
    all_label = []
    for f in os.scandir(top_folder):
        if f.is_dir():
            print(f)
            m, dset, lset = create_image_array_from_a_folder(f.path, word_dict[f.name])
            if len(all_set) == 0:
                print("all set null")
                all_set = dset
                all_label = lset
            else:
                all_set = np.concatenate((all_set, dset), axis=0)
                all_label = np.concatenate((all_label, lset), axis=0)
        print(np.array(all_set).shape)
        print(np.array(all_label).shape)
    num_data = len(all_set)
    datum_shape = all_set[0].shape
    #shuffle
    perm = np.random.permutation(num_data)
    all_set = all_set[perm]
    all_label = all_label[perm]
    hdf_file = h5py.File(target_file, 'w')
    my_chunks = (1, datum_shape[0], datum_shape[1], datum_shape[2])
    hdf_dset = hdf_file.create_dataset(data_set_name, dtype='float', data=all_set, chunks=my_chunks)
    hdf_lset = hdf_file.create_dataset(data_lable_name, data= all_label)
    hdf_file.flush()
    hdf_file.close()
        
create_dataset_from_folder('my_testsets', 'my_testsets/test_signs.h5', 'test_set_x', 'test_set_y') 
create_dataset_from_folder('my_datasets', 'my_datasets/train_signs.h5', 'train_set_x', 'train_set_y')      

img_folder = 'my_test/test_train'
sample_count, dset, lset = create_image_dataset(img_folder, 0.0)
print('count:' + str(sample_count))
print('dataset:'+ str(dset.shape))
print('labelset:'+ str(lset.shape))

print(dset[:, 32, 32, :])
print(lset[:])

sample_count2, dset2, lset2 = create_image_dataset(img_folder, 1.0)
print('count:' + str(sample_count2))
print('dataset:'+ str(dset2.shape))
print('labelset:'+ str(lset2.shape))

print(np.concatenate((lset, lset2, lset), axis=0))

# f = h5py.File('my_datasets/test.h5', 'w')
# train_size = 5
# train = []
# # will be 1200
# dset = f.create_dataset('train_set_x', shape=(train_size, 64, 64, 3))
# lset = f.create_dataset('train_set_y', shape=(train_size, 1))
# file_count = 0
# for filename in listdir(img_folder):
#     cur_file = join(img_folder, filename)
#     print(cur_file)
#     img = image.load_img(cur_file, target_size=(64, 64))
#     x = image.img_to_array(img)
#     train.append(x)
#     dset[file_count] = x
#     lset[file_count] = file_count
#     file_count += 1
# print(dset[:, 32, 32, :])
# print(lset[:])
# print(np.array(train,dtype='float32').shape)
# # work = np.array(dset)
# #shuffle
# perm = np.random.permutation(train_size)
# dset = np.array(dset)[perm]
# lset = np.array(lset)[perm]
# print(dset[:, 32, 32, :])
# print(lset[:])
# f.close()

double_check = h5py.File('my_testsets/test_signs.h5', 'r')
check_x =  np.array(double_check["test_set_x"][:]) # your test set features
check_y =  np.array(double_check["test_set_y"][:])
print(check_x[:, 32, 32, :])
print(check_y[:])
pickA = check_x[333,:]
A_image=image.array_to_img(pickA)
A_label = check_y[333]
double_check.close()




