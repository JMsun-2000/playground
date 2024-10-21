#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:24:21 2021

@author: sunjim
"""
from random import choice
from copy import deepcopy

def readGraph(graph_file):
    graph_dict = {}
    with open(graph_file) as f:
        for line in f:
            elements = [x for x in line.split('\t')][:-1]
            graph_dict[str(elements[0])] = elements[1:]
    return graph_dict

def randomContractOnce(cur_graph):
    u = choice(list(cur_graph.keys()))
    v = choice(cur_graph[u])
    new_key = u+'-'+v
    cur_graph[new_key] = cur_graph[u]+cur_graph[v] 
    del cur_graph[u]
    del cur_graph[v]
    # remove u,v be linked
    for key in cur_graph.keys():
        copy = cur_graph[key][:]
        if new_key == key:
            for item in copy:
                if item == u or item == v:
                    cur_graph[key].remove(item)
        else:
            for item in copy:
                if item == u or item == v:
                    cur_graph[key].remove(item)
                    cur_graph[key].append(new_key)
    

def doMinCut(graph):
    # node number
    n = len(graph)
    # max
    min_cut = n*n
    
    for i in range(n):
        copy = deepcopy(graph)
        while len(copy) > 2:
            randomContractOnce(copy)
        min_cut = min(min_cut, len(list(copy.values())[0]))
    return min_cut

original_graph = readGraph("kargerMinCutClass1-4.txt")

print(doMinCut(original_graph))