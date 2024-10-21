#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:29:42 2021

@author: sunjim
"""

def readJobs(job_file):
    job_list = []
    job_count = 0
    
    with open(job_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0 :
                job_count = int(line)
                print(job_count)
                continue
            
            oneJob = [int(x) for x in line.split()]
            job_list.append(oneJob)        

    return job_list


def sumWeightCompletionTimes(jobs):  
    true_length = 0
    sum_weight = 0
    for j in jobs:
        true_length += j[1]
        sum_weight += true_length * j[0]
        
    return sum_weight


all_jobs = readJobs("jobs_Class3-1.txt")
all_jobs.sort(key=lambda x:((x[0]-x[1]), x[0]), reverse=True)
    
sum_weight_sub = sumWeightCompletionTimes(all_jobs)
# anwser is 69119377652
print('sum of completion times(subtract):', sum_weight_sub)

all_jobs.sort(key=lambda x:(x[0]/x[1]), reverse=True)

sum_weight_ratio = sumWeightCompletionTimes(all_jobs)

# 67311454237
print('sum of completion times(ratio):', sum_weight_ratio)   