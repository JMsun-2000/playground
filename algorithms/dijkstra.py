#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:52:43 2021

@author: sunjim
"""




def readGraph(graph_file):
    vertixes = []
    edges = []
    with open(graph_file) as f:
        for line in f:
            elements = [x for x in line.split('\t')][:-1]
            point1 = elements.pop(0)
            vertixes.append(point1)
            for element in elements:
                values = element.split(',')
                edges.append([point1, values[0], int(values[1])])
            #print (vertixes, edges)
    return vertixes, edges

def Dijkstra(V, all_edges, start='1'):
    A={} # shortest path distances 
    X=[] # processed vertixes
    all_vertix = len(V)
    
    # first step
    X.append(start)
    V.remove(start)
    A[start] = 0 
        
    while len(X) < all_vertix:
        cur_min_distanct = 1000000
        cur_min_edge = (None, None)
        
        # Todo: use heap to manage edges
        for edge in all_edges:
            if (edge[0] in X and edge[1] in V) or (edge[1] in X and edge[0] in V):
                source_x = None
                target_v = None
                if edge[0] in V:
                    target_v = edge[0]
                    source_x = edge[1]
                else:
                    target_v = edge[1]
                    source_x = edge[0]
                if A[source_x] + edge[2] < cur_min_distanct:
                    cur_min_distanct = A[source_x] + edge[2]
                    cur_min_edge = (source_x, target_v)
        
        if cur_min_edge != (None , None):
            X.append(cur_min_edge[1])
            V.remove(cur_min_edge[1])
            A[cur_min_edge[1]] = cur_min_distanct
        else:
            print("warning: there is no path for", V)
            break
    return A
                
            
            
    
vertix, edges = readGraph("dijkstraData_Class2-2.txt")
shortPath = Dijkstra(vertix, edges, '1')
print(shortPath['7'],shortPath['37'],shortPath['59'],shortPath['82'],shortPath['99'],
      shortPath['115'],shortPath['133'],shortPath['165'],shortPath['188'],shortPath['197'])
