#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:55:32 2022

@author: sunjim
"""

from math import sqrt 
from itertools import combinations
import numpy as np
import time


MAX_INF = float('Inf')

def readGraph(edges_file):
    edges = {}
    vertixes = []
    v_with_neg_cost = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [x for x in line.split()]
                print(job_count)
                continue
            
            points = [float(x) for x in line.split()]
            vertixes.append(points)
            
            # debug
            if num == 20:
                break
        
        for i in range(len(vertixes)-1):
            for j in range(i+1, len(vertixes)):
                distance = sqrt(pow(vertixes[i][0] - vertixes[j][0], 2) + pow(vertixes[i][1] - vertixes[j][1], 2))
                edges[(i+1,j+1)] = distance
        
        
            
    #vertixes = dict(sorted(vertixes.items(), key=lambda item: item[0]))
    return edges, vertixes


def fill_A(city_set, city_num, target_c):
     global A
     global all_edges
     set_end_key = (city_set, target_c)
     #print(set_end_key, city_num)
     if set_end_key in A:
         return A[set_end_key]
     
     for m in range(1, city_num):
     #    print("m=", m)
        # print("all_set:", list(combinations(city_set, m)))
         for s in list(combinations(city_set[1:], m)):
        #     print('s=', s)
             for j in s:
              #   print('j=', j)
               #  print("set=",s, j)
                 sub_cities = list((1,)+s)

                 sub_cities.remove(j)
                 cur_min = None
                 sub_cities = tuple(sub_cities)
                 
                 for k in sub_cities:
                     if (sub_cities, k) not in A:
                         continue
                     cur_dist = A[(sub_cities, k)] + all_edges[(min(k,j),max(k,j))]
                     if cur_min == None:
                         cur_min = cur_dist
                     else:
                         cur_min = min(cur_min, cur_dist)
                 if cur_min != None:    
                     A[((1,)+s, j)] = cur_min
    
     return A[set_end_key] if set_end_key in A else None
    

    
def tsp_by_DPA(city_num):
    target_city_list = tuple(range(1, city_num+1))
    
    min_tour = MAX_INF
    for j in range(2, city_num+1):
   #     print('key=',(target_city_list, j))
        full_path = fill_A(target_city_list, city_num, j)
        if full_path == None:
            continue
        cur_dist = full_path + all_edges[(1, j)]
        min_tour = min(min_tour, cur_dist)
    
    return min_tour
        

def better_tsp(city_num, c_edges):
    cache_sets = np.full([1<<(city_num), city_num], None)
    cache_sets[1, 0] = 0
    
    # sets 
    for i in range(1, 1 << city_num):
        # city 
     #   print("set combine:", "{0:b}".format(i))
        for j in range(city_num):
            if cache_sets[i, j] == None:
                continue
            # every bit be set 1, represents a city visited 
            for k in range(1, city_num):
                if (i & (1 <<k)) != 0:
                    # if not zero means some points must have been visited
                    continue
                add_new = (i | (1 << k))
                if cache_sets[add_new, k] == None:
                    # need set
                    cache_sets[add_new, k] = cache_sets[i, j] + c_edges[(min(k+1,j+1),max(k+1,j+1))]
                
                cache_sets[add_new, k] = min(cache_sets[add_new, k], cache_sets[i, j] + c_edges[(min(k+1,j+1),max(k+1,j+1))])
    
    min_tour = MAX_INF
    for i in range(1, city_num):
        cur_dist = cache_sets[(1 << city_num)-1, i]
        if cur_dist != None:
            min_tour = min(min_tour, (cur_dist + c_edges[(1, i+1)]))
            
    return min_tour

def more_better_tsp(city_num, c_edges):
    target_city_num = city_num - 1
    cache_sets = np.full([1<<(target_city_num), target_city_num], None)
    
    for i in range(target_city_num):
        first_stop = 1 << i
        cache_sets[first_stop, 0] = c_edges[(1, i+2)] 
    
    # sets 
    for i in range(1, 1 << target_city_num):
        # city 
      #  print("set combine:", "{0:b}".format(i))
        for j in range(target_city_num):
            if cache_sets[i, j] == None:
                continue
            # every bit be set 1, represents a city visited 
            for k in range(1, target_city_num):
                if (i & (1 <<k)) != 0:
                    # if not zero means some points must have been visited
                    continue
                add_new = (i | (1 << k))
                if cache_sets[add_new, k] == None:
                    # need set
                    cache_sets[add_new, k] = cache_sets[i, j] + c_edges[(min(k+2,j+2),max(k+2,j+2))]
                
                cache_sets[add_new, k] = min(cache_sets[add_new, k], cache_sets[i, j] + c_edges[(min(k+2,j+2),max(k+2,j+2))])
    
    min_tour = MAX_INF
    for i in range(0, target_city_num):
        cur_dist = cache_sets[(1 << target_city_num)-1, i]
        if cur_dist != None:
            min_tour = min(min_tour, (cur_dist + c_edges[(1, i+2)]))
            
    return min_tour
    
                
        

all_edges, list_vertixes = readGraph("tsp_Class4-2.txt")
city_num = len(list_vertixes)
A = {}
A[((1,), 1)] = 0
# Test case TSP 5 min tour: 8387.077130278542
# Test case TSP 6 min tour: 8607.433161541036
# Test case TSP 7 min tour: 9498.200629783674
# Test case TSP 8 min tour: 9765.196115731862
# Test case TSP 9 min tour: 10769.20489976818
# Test case TSP 10 min tour: 12349.980743996226

start_time = time.time()
print("TSP more better min tour:",more_better_tsp(city_num, all_edges))  
print("--- %s seconds ---" % (time.time() - start_time))
# TSP 25 min tour: 26442.73030895475
start_time = time.time()
print("TSP better min tour:",better_tsp(city_num, all_edges))  
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print("TSP best min tour:",tsp_by_DPA(city_num))
print("--- %s seconds ---" % (time.time() - start_time))