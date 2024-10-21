#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:49:48 2021

@author: sunjim
"""
import sys

def readNumbers(seg_file):
    number_array = []
    with open(seg_file) as f:
        for line in f:
            coord = int(line)
            number_array.append(coord)
    return number_array



def sortAndCount(input_array, array_length):
    output_array = []
    revise_count = 0
    if array_length == 1:
        return input_array, revise_count
    else:
        half_length = array_length//2
        sorted_left, revised_left = sortAndCount(input_array[:half_length], half_length)
        sorted_right, revised_right = sortAndCount(input_array[half_length:], (array_length-half_length))
        # merge
        i = 0
        j = 0
        revise_count = revised_left+revised_right
        for k in range(0, array_length):
            if sorted_left[i] < sorted_right[j]:
                output_array.append(sorted_left[i])
                i += 1
            else:
                output_array.append(sorted_right[j])
                j += 1
                revise_count += (len(sorted_left) - i)
            if i==len(sorted_left):
                for v_in_r in sorted_right[j:]:
                    output_array.append(v_in_r)
                break
            if j==len(sorted_right):
                for v_in_l in sorted_left[i:]:
                    output_array.append(v_in_l)
                break
            
                
    return output_array, revise_count
        
    
number_inputs = readNumbers("IntegerArrayClass1-2.txt") 
sorted_result, revised_all = sortAndCount(number_inputs, len(number_inputs))

print(len(sorted_result), revised_all)

for k in range(0, len(sorted_result)-1):
    if sorted_result[k] > sorted_result[k+1]:
        print ("wrong: No.",k," is greater than No.", k+1, "(",sorted_result[k], ">", sorted_result[k+1])
        break
    