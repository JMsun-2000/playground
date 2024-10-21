#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 20:36:41 2021

@author: sunjim
"""

def readGraph(edges_file, digitization=True):
    nodes = {}
    
    with open(edges_file) as f:
        for num, line in enumerate(f, 0):
            # print(num)
            if num == 0:
                node_count, bit_count = [x for x in line.split()]
                print('all nodes count:', node_count, ', present by:',bit_count)
                continue
            
            a = "".join(line.strip().split(" "))
            if digitization:
                a=int(a, 2)
            nodes[a] = True
            
            # debug 
            # if num > 20000:
            #     break
               

    return nodes


    

def nodeSynapse(cell, bit_count=24):
    synapse={}

    for i in range(bit_count):
        dist_1 = cell ^ (1 << i)
        synapse[dist_1]= True
        #print('i=',i, ': ', dist_1)
        for j in range(i+1, bit_count):
            dist_2 = dist_1 ^ (1 << j)
            synapse[dist_2] = True
            #print('ij=',i,j, ': ', dist_2)
    return synapse

def findCluster(all_nodes):
    # create cluster with synapse
    cluster_dict = {}
    for node in all_nodes:
        cluster_dict[(int(node, 2), )] = nodeSynapse(node)
        
    for node in all_nodes:
        node_v = int(node, 2)
        need_merge=[]
        for cluster_key in cluster_dict:  
            if node_v in cluster_key:
                need_merge.append(cluster_key)
                continue
            else:
                # find if the node touch Synapse
                if node_v in cluster_dict[cluster_key]:
                   need_merge.append(cluster_key)
        # do merge
        if len(need_merge) == 1:
            continue
        else:
            new_key=()
            new_synapse={}
            for joint in need_merge:
                new_key += joint
                new_synapse.update(cluster_dict.pop(joint))
            cluster_dict[new_key]=new_synapse
    return cluster_dict

def findClusterSolution2(all_nodes):
    cluster_block = []
    
    while len(all_nodes) > 0:
        top = list(all_nodes.keys())[0]
        all_nodes.pop(top)
        queue = []
        queue.append(top)
        cur_cluster = []
        while len(queue) > 0:
            node = queue.pop(0)
            cur_cluster.append(node)
            synapses = nodeSynapse(node)
            # check if synapse touches nodes
            for syn in synapses:
                if syn in all_nodes:
                    all_nodes.pop(syn)
                    queue.append(syn)
        cluster_block.append(cur_cluster)
        
    return cluster_block
                
        
no_dup_nodes = readGraph('clustering_big_Class3-2.txt')
print(len(no_dup_nodes))    
# 6118    
print('clusters size:', len(findClusterSolution2(no_dup_nodes)))  


















      
            
            
    
    