#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:17:00 2021

@author: sunjim
"""

def readGraph(edges_file):
    edges = {}
    vertixes = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [x for x in line.split()]
                print(job_count)
                continue
            
            edge = [int(x) for x in line.split()]
            vertixes[edge[0]] = True
            vertixes[edge[1]] = True
            edges[(min(edge[0], edge[1]), max(edge[0], edge[1]))] = edge[2]       

    return edges, vertixes


def primMST(all_edges, all_vertixes):
    MSTree = {}
    X={}
    V = all_vertixes.copy()
    X[1] = True
    del V[1]
    
    while len(X) != len(all_vertixes):
        cheapest = None
        for e in all_edges:
            if ((e[0] in X) and (e[1] in V)) or ((e[1] in X) and (e[0] in V)):
                if cheapest == None:
                    cheapest = e
                else:
                    cheapest = e if all_edges[e] < all_edges[cheapest] else cheapest
                    
        if cheapest == None:
            print("break path graph!")
            break
        else:
            MSTree[cheapest] = all_edges[cheapest]
            if cheapest[0] in X:
                X[cheapest[1]] = True
                del V[cheapest[1]]
            else:
                X[cheapest[0]] = True
                del V[cheapest[0]]
    return MSTree

list_edges, list_vertixes = readGraph("edges_Class3-1.txt")
mst = primMST(list_edges, list_vertixes)
print('MST cost:', sum(mst.values()))

    
    