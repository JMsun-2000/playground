#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:43:30 2021

@author: sunjim
"""

def readGraph(edges_file):
    edges = {}
    cluester = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [x for x in line.split()]
                print(job_count)
                continue
            
            edge = [int(x) for x in line.split()]
            cluester[(edge[0], 0)] = True
            cluester[(edge[1], 0)] = True
            edges[(min(edge[0], edge[1]), max(edge[0], edge[1]))] = edge[2]       

    return edges, cluester



all_edges, begin_cluester = readGraph("clustering1_Class3-2.txt")

def cluester_merge(list_edges, cur_cluester, parts=4):
    sorted_edges = {k: v for k, v in sorted(list_edges.items(), key=lambda item: item[1])}
    max_sapce = list(sorted_edges.values())[-1]
    
    for edge in sorted_edges:
        #print (edge)
        vertix_p = edge[0]
        vertix_q = edge[1]
        found_p = None
        found_q = None
        for culester_key in cur_cluester:
            if found_p == None and vertix_p in culester_key:
                found_p = culester_key
            if found_q == None and vertix_q in culester_key:
                found_q = culester_key
            if found_p != None and found_q != None:
                break
            
        if found_p == found_q:
            # already in same cluester
            continue
        else:
            if len(cur_cluester) > parts:
                # merge
                del cur_cluester[found_p]
                del cur_cluester[found_q]
                new_key = found_p + found_q
                cur_cluester[new_key] = True
            else:
                if max_sapce > sorted_edges[edge]:
                    max_sapce = sorted_edges[edge]
    return max_sapce, len(cur_cluester)

max_clustering, final_parts = cluester_merge(all_edges, begin_cluester)
# should get 106
print('cluster-', final_parts, ' maximum spacing ', max_clustering)
            
        
        

    
    