#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:50:24 2021

@author: sunjim
"""

def readNumber(number_file):
    positive_number_hash = {}
    negative_number_hash = {}
    
    with open(number_file) as f:
        m = 0
        for line in f:
            element = [x for x in line.split()][0]
            if int(element) >= 0:
                positive_number_hash[int(element)] = True
            else:
                negative_number_hash[int(element)] = True 
            m += 1
    print('input:', m, 'pos:', len(positive_number_hash), 'neg:', len(negative_number_hash))
    return positive_number_hash, negative_number_hash


pos_input, neg_input = readNumber("algo1-programming_prob-2sum_Class2-4.txt")
found_cnt = {}

for x in pos_input:
    for t in range(-10000, 10001):
        # print(x)
        potential_key = t-x
        if potential_key in neg_input:
            found_cnt[t] = True
            
print ('found:', len(found_cnt))