#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:04:13 2022

@author: sunjim
"""
from math import sqrt 
import time


MAX_INF = float('Inf')

def readGraph(edges_file):
    edges = {}
    vertixes = {}
    v_with_neg_cost = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [x for x in line.split()]
                print(job_count)
                continue
            
            points = [float(x) for x in line.split()]
            vertixes[int(points[0])] = (points[1], points[2])
            
            #debug
            # if num == 20:
            #     break
        
        # for i in vertixes:
        #     for j in vertixes:
        #         if i == j: continue
        #         distance_sqr = pow(vertixes[i][0] - vertixes[j][0], 2) + pow(vertixes[i][1] - vertixes[j][1], 2)
        #         if i in edges:
        #             edges[i].append([j, distance_sqr])
        #         else:
        #             edges[i] = [j, distance_sqr]
                
    return vertixes


def heuristic_TSP(original_vertixes):
    vertixes = original_vertixes.copy()
    target_city = 1
    cities_count = len(vertixes)
    visited_cities = []
    min_tour = 0
    cur_city = vertixes.pop(target_city)
    visited_cities.append(target_city)
    print('start:', target_city)
    
    while len(visited_cities) < cities_count:   
        # looking for nearest
        nearest_sqr = MAX_INF
        for c in vertixes:
            distance_sqr = pow(cur_city[0] - vertixes[c][0], 2) + pow(cur_city[1] - vertixes[c][1], 2)
            if distance_sqr < nearest_sqr:
                nearest_sqr = distance_sqr
                target_city = c
        
       # print('nearest:', sqrt(nearest_sqr))
        cur_city = vertixes.pop(target_city)
        visited_cities.append(target_city)
        print('visited:', target_city)
        min_tour += sqrt(nearest_sqr)
        
    # add back 
    min_tour += sqrt(pow(original_vertixes[1][0] - original_vertixes[target_city][0], 2) + pow(original_vertixes[1][1] - original_vertixes[target_city][1], 2))
 
    return min_tour, visited_cities
                
start_time = time.time()
list_vertixes = readGraph("nn_Class4-3.txt")
tour_distance, visted_order = heuristic_TSP(list_vertixes)
print('distance:', tour_distance)
# print('visited order:', visted_order)
print("-all-- %s seconds ---" % (time.time() - start_time))

# debug double check
# check_distance = 0
# for i in range(len(visted_order)):
#     cur_d = 0
#     if i != len(visted_order)-1:
#         cur_d = sqrt(pow(list_vertixes[visted_order[i]][0] - list_vertixes[visted_order[i+1]][0], 2) + pow(list_vertixes[visted_order[i]][1] - list_vertixes[visted_order[i+1]][1], 2))
#     else:
#         cur_d = sqrt(pow(list_vertixes[visted_order[i]][0] - list_vertixes[visted_order[0]][0], 2) + pow(list_vertixes[visted_order[i]][1] - list_vertixes[visted_order[0]][1], 2))
#     print("check:", cur_d)
#     check_distance += cur_d

# print('dobule check:', check_distance)
    

