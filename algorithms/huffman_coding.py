#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 19:50:32 2021

@author: sunjim
"""

def readJobs(job_file):
    job_list = {}
    job_count = 0
    
    with open(job_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0 :
                job_count = int(line)
                print(job_count)
                continue
            
            oneJob = int(line)
            job_list[num] = oneJob     

    return job_list

def huffmanCode(char_list):
    while len(char_list) > 1:
        char_list = dict(sorted(char_list.items(), key=lambda item: item[1]))
        
        # pick 2 smallest
        first_key = next(iter(char_list))
        first_value = char_list.pop(first_key)
        second_key = next(iter(char_list))
        second_value = char_list.pop(second_key)
        char_list[(first_key, second_key)] = first_value + second_value          
    
    return next(iter(char_list))

def miniTree(tree_root):
    
    if type(tree_root) is int:
        return 0
    return min(miniTree(tree_root[0]), miniTree(tree_root[1])) + 1

def maxiTree(tree_root):
    
    if type(tree_root) is int:
        return 0
    return max(maxiTree(tree_root[0]), maxiTree(tree_root[1])) + 1
    
    
    

all_char = readJobs("huffman_Class3-3.txt")
tree_char = huffmanCode(all_char)
print(miniTree(tree_char)) # 9
print(maxiTree(tree_char)) # 19


