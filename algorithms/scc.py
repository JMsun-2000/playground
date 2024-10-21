#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:15:46 2021

@author: sunjim
"""


# from sys import setrecursionlimit
# from threading import stack_size
# setrecursionlimit(60000)
# stack_size(67108864)



n = 875714


def readGraph(graph_file, is_rev=False, label_f=None):
    graph_dict = {}
    test = 0
    with open(graph_file) as f:
        for line in f:
            elements = [x for x in line.split()]
            if is_rev:
                vertex_key = elements[1] if label_f == None else label_f[elements[1]]
                point_to = elements[0] if label_f == None else label_f[elements[0]]
            else:
                vertex_key = elements[0] if label_f == None else label_f[elements[0]]
                point_to = elements[1] if label_f == None else label_f[elements[1]]
            if vertex_key in graph_dict:
                graph_dict[vertex_key].append(point_to)
            else:
                graph_dict[vertex_key] = [point_to]    
        
            test += 1
         
    print('read lines',test)        
    return graph_dict



def DFS_Loop(graph):
    global s
    global visited
    for i in range(n, 0, -1):
        if not str(i) in visited:
            s = str(i)
            DFS(graph, str(i))
            
     
def DFS(graph, node_i):
    global finishing_times
    global s
    global fun_to_times
    global leader
    global visited
    
    stack = []
    stack.append(node_i)
    
    while len(stack) > 0:
        node = stack[-1]
        if not node in visited:
            visited[node] = True
            leader[node] = s
            
        remove_from_stack = True
        if node in graph:
            for j in graph[node]:
                if not j in visited:
                    stack.append(j)
                    remove_from_stack = False
                    break
        if remove_from_stack:
            finishing_times += 1
            fun_to_times[node] = str(finishing_times)
            stack.pop()
    # while len(stack) > 0:
    #     cur_vertex, phase = stack.pop()
    #     if phase == 1:
    #         visited[str(cur_vertex)] = True
    #         leader[str(cur_vertex)] = s
    #         next_found = False

    #         if str(cur_vertex) in graph:
    #             for j in graph[str(cur_vertex)]:
    #                 if not str(j) in visited:
    #                     stack.append((cur_vertex, 1))
    #                     stack.append((j, 1))
    #                     next_found = True
    #                     break
                    
    #         if next_found == False:
    #             stack.append((cur_vertex, 2))
    #     if phase == 2:
    #         finishing_times += 1
    #         fun_to_times[str(node_i)] = str(finishing_times)
    

original_rev_graph = readGraph("SCC_Class2-1.txt", True, None)
s = None
visited = {}
leader = {}
finishing_times = 0 
fun_to_times = {}      
DFS_Loop(original_rev_graph)
label_graph = readGraph("SCC_Class2-1.txt", False, fun_to_times)
s = None
visited = {}
leader = {}
finishing_times = 0 
fun_to_times = {}  
DFS_Loop(label_graph)

sccs = {}
for i in leader:
  if leader[i] not in sccs:
    sccs[leader[i]] = 1
  else:
    sccs[leader[i]] +=1
 
sorted_order = sorted(sccs.items(), key=lambda x: x[1], reverse=True)