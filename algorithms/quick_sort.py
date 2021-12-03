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
    
def quickSortLast(array_input, l, r):
    if l == r:
        return 0
    else:
        # use last
        m_cur = r-l-1
        # exchange swap firt and last
        array_input[l], array_input[r-1] = array_input[r-1], array_input[l]
        piovt_p = partition(array_input, l, r)
        m_left = quickSortLast(array_input, l, piovt_p)
        m_right = quickSortLast(array_input, piovt_p+1, r)
        return m_cur + m_left+ m_right
    
def quickSortMiddle(array_input, l, r):
    if l == r:
        return 0
    else:
        # use middle
        first = array_input[l]
        last = array_input[r-1]
        middle_index = l+(r-1-l)//2
        middle = array_input[middle_index]
        
        if (middle > last and last > first) or (middle < last and last < first):
            # last is middle
            array_input[l]=last
            array_input[r-1] = first
        elif (first > middle and middle > last) or (first < middle and middle < last):
            # middle is middle
            array_input[l]=middle
            array_input[middle_index] = first

        
        m_cur = r-l-1
        piovt_p = partition(array_input, l, r)
        m_left = quickSortMiddle(array_input, l, piovt_p)
        m_right = quickSortMiddle(array_input, piovt_p+1, r)
        return m_cur + m_left+ m_right


number_inputs = readNumbers("QuickSortClass1-3.txt") 
m_all = quickSortFirst(number_inputs, 0, len(number_inputs))
# should be 162085
print('first compared count:', m_all)

number_inputs = readNumbers("QuickSortClass1-3.txt") 
m_all = quickSortLast(number_inputs, 0, len(number_inputs))
# should be 164123
print('last compared count:', m_all)

number_inputs = readNumbers("QuickSortClass1-3.txt") 
m_all = quickSortMiddle(number_inputs, 0, len(number_inputs))
# should be 
print('middle compared count:', m_all)

for k in range(0, len(number_inputs)-1):
    if number_inputs[k] > number_inputs[k+1]:
        print ("wrong: No.",k," is greater than No.", k+1, "(",number_inputs[k], ">", number_inputs[k+1])
        break