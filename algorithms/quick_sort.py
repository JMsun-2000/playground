#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:46:38 2021

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



def partition(array_for_sort, l, r):
    p = array_for_sort[l]
    i = l+1
    
    for j in range(l+1, r):
        if array_for_sort[j] < p:
            array_for_sort[i], array_for_sort[j] = array_for_sort[j], array_for_sort[i]
            i += 1
    
    array_for_sort[l], array_for_sort[i-1] = array_for_sort[i-1], array_for_sort[l]
    
    return i-1

def quickSortFirst(array_input, l, r):

    if l == r:
        return 0
    else:
        # use first
        m_cur = r-l-1
        piovt_p = partition(array_input, l, r)
        m_left = quickSortFirst(array_input, l, piovt_p)
        m_right = quickSortFirst(array_input, piovt_p+1, r)
        return m_cur + m_left+ m_right

number_inputs = readNumbers("QuickSortClass1-3.txt") 
m_all = quickSortFirst(number_inputs, 0, len(number_inputs))

print('compared count:', m_all)

for k in range(0, len(number_inputs)-1):
    if number_inputs[k] > number_inputs[k+1]:
        print ("wrong: No.",k," is greater than No.", k+1, "(",number_inputs[k], ">", number_inputs[k+1])
        break