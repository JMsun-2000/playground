#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:57:09 2022

@author: sunjim
"""

def readGraph(edges_file):
    vertixes = {}
    v_with_neg_cost = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                job_count = [x for x in line.split()]
                print(job_count)
                continue
            
            # v1, v2 = map(int, line.split())
            # print(v1, v2)
            points = [int(x) for x in line.split()]
            vertixes[(points[0], points[1])] = True
            
            #debug
            # if num == 200:
            #     break
            
    return vertixes


def dfs(graph, s, visited, st):
    stack = []
    stack.append(s)
    
    while len(stack) > 0:
        node = stack[-1]
        if not node in visited:
            visited.add(node)
            
        remove_from_stack = True
        if node in graph:
            for j in graph[node]:
                if not j in visited:
                    stack.append(j)
                    remove_from_stack = False
                    break
        if remove_from_stack:
            st.append(node)
            stack.pop()

'''
 ( x V y),  x=False 必须 y = true, 直接符号化(fasleX->trueY),才能满足set。写作  ~x -> y.  
 (~x V y), x=True  必须 y = true,   x -> y
 (x V -y), x=False  必须 y = False,   ~x -> ~y
 (~x V -y), x=Ture  必须 y = False,   x -> ~y
发现规律，（A V B）必须 ~A -> B. 或者 ~B -> A

通过 SCC寻找闭环，如果发现 X ~> ~X 且 ~X ~> X, 矛盾，不存在 satisfiable
'''

def isSatisfiable(vsets_dict):
    sat = True
    sat2Graph = {}
    for vset in vsets_dict:
        v1, v2 = vset
        
        if -v1 in sat2Graph:
            sat2Graph[-v1].append(v2)
        else:
            sat2Graph[-v1] = [v2] 
        
        if -v2 in sat2Graph:
            sat2Graph[-v2].append(v1)
        else:
            sat2Graph[-v2] = [v1]
            
    print('sat2Graph len', len(sat2Graph))
            
    # SCC 
    visited = set()
    finish_times_order = []
    # found leader
    for v in sat2Graph:
        if v not in visited:
            dfs(sat2Graph, v, visited, finish_times_order)
        
    # reverse graph
    rev_graph = {}
    for v_s in sat2Graph:
        for v_t in sat2Graph[v_s]:
            if v_t in rev_graph:
                rev_graph[v_t].append(v_s)
            else:
                rev_graph[v_t]=[v_s]
    
    # dfs again with last finish_times_order, last is the biggist leader
    # reset visitied
    visited = set()
    while len(finish_times_order)>0:
        latent_leader = finish_times_order.pop()
        if latent_leader not in visited:
            # arrived here, you are a real leader
            scc = []
            dfs(rev_graph, latent_leader, visited, scc)
            scc_set = set(scc)
            for v in scc:
                if -v in scc:
                    # cycle must make v ~> -v, and -v ~> v
                    print(v, " and ", -v, " not satisfable")
                    sat = False
                    break
    
    return sat
        
        
        
FILE_NAMES = ["2sat1_Class4-4.txt","2sat2_Class4-4.txt","2sat3_Class4-4.txt","2sat4_Class4-4.txt","2sat5_Class4-4.txt","2sat6_Class4-4.txt"]

cur_pick = 5
list_vset = readGraph(FILE_NAMES[cur_pick])
print(FILE_NAMES[cur_pick], "is", isSatisfiable(list_vset))



# list_vset1 = readGraph("2sat1_Class4-4.txt")   # True
# list_vset2 = readGraph("2sat2_Class4-4.txt")   # False
# list_vset3 = readGraph("2sat3_Class4-4.txt")   # True
# list_vset4 = readGraph("2sat4_Class4-4.txt")   # True
# list_vset5 = readGraph("2sat5_Class4-4.txt")   # False
# list_vset6 = readGraph("2sat6_Class4-4.txt")   # False


        