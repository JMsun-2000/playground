#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:21:11 2021

@author: sunjim
"""

def readItems(items_file):
    items = {}
    knapsack_size = 0
    number_of_items = 0
    
    with open(items_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [int(x) for x in line.split()]
                knapsack_size = job_count[0]
                number_of_items = job_count[1]
                print('pack size:', knapsack_size, 'total items:', number_of_items)
                continue
            
            one_item = [int(x) for x in line.split()]
            items[num] = (one_item[0], one_item[1])

    return knapsack_size, items

def readItemsMoreWithSorted(items_file):
    items = []
    knapsack_size = 0
    number_of_items = 0
    sorted_items={}
    unique_weight = {}
    
    with open(items_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [int(x) for x in line.split()]
                knapsack_size = job_count[0]
                number_of_items = job_count[1]
                print('pack size:', knapsack_size, 'total items:', number_of_items)
                continue
            
            one_item = [int(x) for x in line.split()]
            items.append((one_item[0], one_item[1]))
            unique_weight[one_item[1]] = True
    
    items.sort(key=lambda x:(x[1]), reverse=False)
    min_weight = items[0][1]
    min_wgap = knapsack_size
    for i in range(len(items)):
        sorted_items[(i+1)] = items[i]

        

    return knapsack_size, sorted_items, min_weight, number_of_items


def maxValueOfPack(pack_size, available_items):
    item_count = len(available_items)

    pre_A=[0]*(pack_size + 1)
    
    for i in range(1, item_count+1):
        cur_A = pre_A.copy()
        for x in range(0, pack_size+1):
            wi = available_items[i][1]
            vi = available_items[i][0]
            
            if wi > x:
                continue
            cur_A[x] = max(pre_A[x], pre_A[x-wi]+vi)
        pre_A = cur_A
    return pre_A[pack_size]
        

pack_size, all_items = readItems("knapsack1_Class3-4.txt")
# should get 2493893
#print(maxValueOfPack(pack_size, all_items))

def maxValueOfBigBigPack(pack_size, sorted_w_items, min_item_weight):
    item_count = len(sorted_w_items)

    pre_A=[0]*(pack_size + 1)
    
    for i in range(1, item_count+1):
        cur_A = pre_A.copy()
        
        x = min_item_weight-1
        while (x <= pack_size):  #200000
            wi = sorted_w_items[i][1]
            vi = sorted_w_items[i][0]
            
            if wi > x:
                x+= (wi-x)
                continue
            
            cur_A[x] = max(pre_A[x], pre_A[x-wi]+vi)
            x+=1
            
        pre_A = cur_A
    return pre_A[pack_size]



def knapsackBig(num, size):
    global cache_result
    global sorted_items
    if num == 0 or size == 0:
        return 0
    elif cache_result.get((num,size)) != None:
        return cache_result[(num,size)]
    else:
        if size < sorted_items[num][1]:
            return knapsackBig(num-1 , size)
        else:
            best = max(knapsackBig(num-1,size),knapsackBig(num-1,size-sorted_items[num][1]) + sorted_items[num][0])
            cache_result[(num,size)] = best
            return best

cache_result = {}
#test
big_pack_size, sorted_items, min_weight, number_of_items = readItemsMoreWithSorted("knapsack1_Class3-4.txt")
#big_pack_size, sorted_items, min_weight, number_of_items   = readItemsMoreWithSorted("knapsack_big_Class3-4.txt")
# for i in range(big_pack_size+1):
#     cache_result[(0,i)] = 0

sorted_items[0]=(0, 0)
print(knapsackBig(number_of_items, big_pack_size))




















