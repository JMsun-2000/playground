#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 15:34:34 2022

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
            edges[(edge[0], edge[1])] = edge[2]    
            
    vertixes = dict(sorted(vertixes.items(), key=lambda item: item[0]))
    return edges, vertixes

def Floyd_Warshall(vertixes, edges):
    A = {}
    N = len(vertixes)
    # initialize
    for i in range(1, N+1):
        A[(i, i, 0)] = 0
    for e in edges:
        A[e+(0,)] = edges[e]
    
    #åprint(A)
    
    min_path = 1000000000000000
    #print(A)
    for k in range(1, N+1):
        print(k)
        for i in range(1, N+1):
            for j in range(1, N+1):
                # None key means no path, infinite number
                if (i,j,k-1) in A:
                    if ((i, k, k-1) in A) and ((k, j, k-1) in A):
                        A[(i,j,k)] = min(A[(i,j,k-1)], (A[(i,k,k-1)] + A[(k,j,k-1)]))
                    else:
                        A[(i,j,k)] = A[(i,j,k-1)]
                else:
                    if ((i, k, k-1) in A) and ((k, j, k-1) in A):
                        A[(i,j,k)] = A[(i,k,k-1)] + A[(k,j,k-1)]
                # debug
                if (i,j,k) in A:
                    #print("A(",i,",",j,",",k,")=", A[(i,j,k)])
                    if A[(i,j,k)] < min_path:
                        min_path = A[(i,j,k)]
                        print ('min_path', min_path)
                        # check negative cycle
                        if i==j and A[(i,j,k)] < 0:
                            print("warning: negative cycle")
                            return None
                
    
    return min_path
    
    


list_edges1, list_vertixes1 = readGraph("g3_Class4-1.txt")
#print(len(list_vertixes), len(list_edges))
print("min g1", Floyd_Warshall(list_vertixes1, list_edges1))
      
# list_edges2, list_vertixes2 = readGraph("g2_Class4-1.txt")
# print("min g2", Floyd_Warshall(list_vertixes2, list_edges2))

# list_edges3, list_vertixes3 = readGraph("g3_Class4-1.txt")
# print("min g3", Floyd_Warshall(list_vertixes3, list_edges3))