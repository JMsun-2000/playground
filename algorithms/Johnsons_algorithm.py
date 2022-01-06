#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:54:52 2022

@author: sunjim
"""
import heapq

MAX_INT = float('Inf')

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
            
            edge = [int(x) for x in line.split()]
            vertixes[edge[0]] = True
            vertixes[edge[1]] = True
            edges[(edge[0], edge[1])] = edge[2]
            if edge[2] < 0:
               v_with_neg_cost[edge[0]] = True
            
    vertixes = dict(sorted(vertixes.items(), key=lambda item: item[0]))
    return edges, vertixes, v_with_neg_cost

# Dijkstra algorithm
def dijkstra_use_heap(adjList, verticesCount, src):
    shortestPaths = [MAX_INT for i in range(verticesCount)]
    visited = [False for i in range(verticesCount)]
    pq = [(0, src)]
    while len(pq) > 0:
        w, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        shortestPaths[u] = w
        for v, wt in adjList[u]:
            if not visited[v]:
                heapq.heappush(pq, (w+wt, v))
    return shortestPaths

def Dijkstra(vertixes, sorted_rweight_edges, start='1'):
    V = list(vertixes.keys())
    A={} # shortest path distances 
    X=[] # processed vertixes
    all_vertix = len(V)
    all_edges = sorted_rweight_edges.copy()
    
    # first step
    X.append(start)
    V.remove(start)
    A[start] = 0 
        
    while len(X) < all_vertix:
        cur_min_distanct = 100000000
        cur_min_edge = (None, None)
        
        # for x in X:
        #     for v in V:
        #        if (x, v) in rweight_edges:
        #            l = A[x] + rweight_edges[(x, v)]
        #            if l < cur_min_distanct:
        #                cur_min_distanct = l
        #                cur_min_edge = (x, v)
               
        
        #Todo: use heap to manage edges
        for edge in all_edges:
            if (edge[0] in X and edge[1] in V):
                source_x = edge[0]
                target_v = edge[1]
                
                if A[source_x] + all_edges[edge] < cur_min_distanct:
                    cur_min_distanct = all_edges[edge] + A[source_x]
                    cur_min_edge = (source_x, target_v)
                    break
        
        if cur_min_edge != (None , None):
            X.append(cur_min_edge[1])
            V.remove(cur_min_edge[1])
            A[cur_min_edge[1]] = cur_min_distanct
            del all_edges[cur_min_edge]
        else:
            print("warning: there is no path for", V)
            break
    return A

def Bellman_Ford(vertixes, edges, start=0):
    target_edges = {}
    default_min = 1000000000
    
    for e in edges:
        if e[1] in target_edges:
            target_edges[e[1]].append((e[0], edges[e]))
        else:
            target_edges[e[1]] = [(e[0], edges[e])]
    
    A={}
    A[0] = 0
    N = len(vertixes)
    for i in range(1, N):
        A[i] = default_min
        
    # print(A)
    perv_A = {}

    for i in range(1, N+1):
        perv_A = A.copy()
        for v in vertixes:
            min_new_e = default_min
            if v in target_edges:
                for src in target_edges[v]:
                    w = src[0]
                    new_value = perv_A[w] + src[1]
                    if min_new_e > new_value:
                        min_new_e = new_value
                A[v] = min(perv_A[v], min_new_e)
    
    return perv_A != A, A
            
                    
    
    
def JohnsonAlgorithm(vertixes, edges, only_check=None):
    # check negative cycle first 
    Gplus = edges.copy()
    Vplus = vertixes.copy()
    add_start = 0
    Vplus[add_start] = True
    for i in range(1, len(Vplus)):
        Gplus[(add_start, i)] = 0
    has_neg_cycle, bellman_A = Bellman_Ford(Vplus, Gplus)
    #print(bellman_A)
    if has_neg_cycle:      
        print("Exit: negative cycle")
        return bellman_A
    print("no negative cycle, A[0], A[1], A[100]", bellman_A[0], bellman_A[1], bellman_A[100])
   
    # reweight edges
    rweight_edges = {}
    for e in edges:
        rweight_edges[e] = edges[e] + bellman_A[e[0]] - bellman_A[e[1]]
        
    rweight_edges = dict(sorted(rweight_edges.items(), key=lambda item: item[1]))
    # print(rweight_edges)
    # return 
    
    min_path = 10000000000
    check_vertixes = only_check if only_check != None else vertixes
    for u in check_vertixes:
        cur_min = 10000000000
        reweight_spath = Dijkstra(vertixes, rweight_edges, u)
        for v in reweight_spath:
            real_cost = reweight_spath[v] - bellman_A[u] + bellman_A[v]
            cur_min = min(real_cost, cur_min)
        min_path = min(min_path, cur_min)
        print('start vertixe: ', u, " shortest path:", cur_min, " shortest in all:", min_path)
        
    return min_path
    
    
    
    
    
    
    
    
    
    

list_edges1, list_vertixes1, need_check = readGraph("g3_Class4-1.txt")


#list_edges1, list_vertixes1 = readGraph("large_Class4-1.txt")
#print(len(list_vertixes), len(list_edges))
print("legal Graph", JohnsonAlgorithm(list_vertixes1, list_edges1, need_check))