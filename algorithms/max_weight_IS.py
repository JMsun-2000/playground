#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 20:39:24 2021

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



def maxWeightIS(vertices_list):
    A = []
    A.append(0)
    A.append(vertices_list[1])
    
    for i in range(2, 1001):
        A.append(max(A[i-1], A[i-2]+vertices_list[i]))
        
    return A
        
def reconstructionMaxWIS(mwis_array, vertices_list):
    S={}
    i = len(mwis_array) - 1
    while i >=1:
        if mwis_array[i-1] >= mwis_array[i-2]+vertices_list[i]:
            if i in (1, 2, 3, 4, 17, 117, 517, 997):
                print (i, '=0')
            i -= 1
        else:
            if i in (1, 2, 3, 4, 17, 117, 517, 997):
                print (i, '=1')
            S[i]=True
            i -= 2
    return S
        
    

all_vertices = readJobs("mwis_Class3-3.txt")
max_wis_array = maxWeightIS(all_vertices)
#10100110  (1, 3, 117, 517)
optimal_selected = reconstructionMaxWIS(max_wis_array, all_vertices)
