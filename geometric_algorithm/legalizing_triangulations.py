#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:19:20 2021

@author: sunjim
"""

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from random import randrange
from sortedcontainers import SortedList, SortedKeyList
from numpy import sqrt
import collections
import copy

POINTS_INDEX = []
CUR_TRIANGLES = {}

class Segment:

    def __init__(self, start, end, m, b):
        
        self.start = start
        self.end = end
        if start is None or end is None:
            self.m = m
            self.b = b
        else:
            self.m = (end[1] - start[1]) / (end[0] - start[0])
            self.b = start[1] - self.m * start[0]
    
    # def __init__(self):
    #     return
        
    def bisector(self):
        mid_point = ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)
        # vertical slope
        v_m = (-1) / self.m
        v_b = mid_point[1] - v_m * mid_point[0]
        return Segment(None, None, v_m, v_b)
    
    def intersect(self, seg):
        x = (seg.b - self.b) / (self.m - seg.m)
        y = seg.m * x + seg.b
        return (x, y)

class Circle:
    def __init__(self, point1, point2, point3):
        # avoid zero division which caused by vertical line
        p1 = point1
        p2 = point2
        p3 = point3
        
        if point1[1] == point2[1]:
            p2 = point3
            p3 = point2
        elif point2[1] == point2[1]:
            p1 = point2
            p2 = point1
        
        s1 = Segment(p1, p2, None, None)
        s2 = Segment(p2, p3, None, None)
        
        print ("s1", s1.start, s1.end, s1.m, s1.b)
        print ("s2", s2.start, s2.end, s2.m, s2.b)
        self.center = s1.bisector().intersect(s2.bisector())
        self.radius_pow = pow((self.center[0] - point1[0]), 2) + pow((self.center[1] - point1[1]), 2)
        self.radius = sqrt(pow((self.center[0] - point1[0]), 2) + pow((self.center[1] - point1[1]), 2))
        
    def isInsideQuick(self, point):
        return
        
    def isInside(self, point):
        print ("check inside", point, "in circle", self.center, self.radius) 
        print (pow((point[0] - self.center[0]), 2), pow((point[1] - self.center[1]), 2), self.radius_pow)
        return pow((point[0] - self.center[0]), 2) + pow((point[1] - self.center[1]), 2) < self.radius_pow

class Edge:
    def __init__(self, point1, point2):
        self.low = min(point1, point2)
        self.high = max(point1, point2)
        self.key = (self.low, self.high)
        self.joints = []
        
    def add_joint(self, point):
        if not point in self.joints:
            self.joints.append(point)

        
    
    '''
    Recall that an edge is illegal if flipping it would increase the angle vector of the adjacent triangles. 
    Equivalently, an edge (a,b) with adjacent triangles is (a,b,c) and (a,b,d) is illegal 
    if c is in the interior of the circle that passes through a, b and d. 
    '''
    def isLegal(self):
        if len(self.joints) < 2:
            return True
        # print(self.low, POINTS_INDEX[self.low], self.high, POINTS_INDEX[self.high])
        # print(self.joints[0], POINTS_INDEX[self.joints[0]], self.joints[1], POINTS_INDEX[self.joints[1]])
        circle1 = Circle(POINTS_INDEX[self.low], POINTS_INDEX[self.high], POINTS_INDEX[min(self.joints[0], self.joints[1])])
        if circle1.isInside(POINTS_INDEX[max(self.joints[0], self.joints[1])]):
            return False
        # circle2 = Circle(POINTS_INDEX[self.low], POINTS_INDEX[self.high], POINTS_INDEX[self.joints[1]])
        # if circle1.isInside(POINTS_INDEX[self.joints[0]]):
        #     return False
        
        return True
    
    def flip(self):
        if len(self.joints) < 2:
            return
        new_low = min(self.joints[0], self.joints[1])
        new_high = max(self.joints[0], self.joints[1])
        self.joints = [self.low, self.high]
        self.low = new_low
        self.high = new_high
        self.key = (new_low, new_high)

def readTriangle(triangulation_file):
    triangle_array = []

    with open(triangulation_file) as f:
        content = f.readlines()
        n, m = content[0].split()
        n = int(n)
        m = int(m)
        
        content_start=1
        #print (content[content_start:content_start+n])
        # read points
        for line in content[content_start:content_start+n]:
            index, x, y = line.split()
            POINTS_INDEX.append((int(x), int(y)))
            
        # read edges
        #print (content[content_start+n: content_start+n+m])
        for line in content[content_start+n: content_start+n+m]:
            a, b, c = line.split()
            triangle_array.append([int(a), int(b), int(c)])

    return triangle_array

def drawtriangles(triangles, debug_mode=False):
    image = Image.new('RGB', (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    count = 0
    
    # draw.polygon([points[0], points[7], points[1]], outline=(randrange(0, 255), randrange(0, 255), randrange(0, 255)))
    # draw.polygon([points[1], points[0], points[3]], outline=(randrange(0, 255), randrange(0, 255), randrange(0, 255)))
    for tri in triangles:
       # draw.polygon([POINTS_INDEX[tri[0]], POINTS_INDEX[tri[1]], POINTS_INDEX[tri[2]]], outline=(randrange(0, 255), randrange(0, 255), randrange(0, 255)))
        draw.polygon([POINTS_INDEX[tri[0]], POINTS_INDEX[tri[1]], POINTS_INDEX[tri[2]]], outline="red")
      
        count +=1
        if debug_mode:
            tri_circle = Circle(POINTS_INDEX[tri[0]], POINTS_INDEX[tri[1]], POINTS_INDEX[tri[2]])
            print ("draw circle", tri_circle.center, tri_circle.radius)
            draw.ellipse((tri_circle.center[0]-tri_circle.radius, tri_circle.center[1]-tri_circle.radius,
                          tri_circle.center[0]+tri_circle.radius, tri_circle.center[1]+tri_circle.radius), outline='blue')
        # debug
        # if count > 0:
        #     break
    
    del draw
    plt.imshow(image)
    plt.show()
    
def insert_edge(p1, p2, p3, result_dict):
    cur_key = (min(p1, p2), max(p1, p2))
    if cur_key in result_dict:
        result_dict[cur_key].add_joint(p3)
    else:
        new_edge = Edge(p1, p2)
        new_edge.add_joint(p3)
        result_dict[cur_key] = new_edge

def collectEdgesFromTriangle(tri, result_dict):
    insert_edge(tri[0], tri[1], tri[2], result_dict)
    insert_edge(tri[0], tri[2], tri[1], result_dict)
    insert_edge(tri[2], tri[1], tri[0], result_dict)
    
def update_edge(begin_point, end_point, update_to, need_replace, edge_checked, not_checked):
    min_idx = min(begin_point, end_point)
    max_idx = max(begin_point, end_point)
    
    cur_key = (min_idx, max_idx)
    
    print ("going to change ", cur_key, "point from ", need_replace, "to ", update_to)
    
    if cur_key in edge_checked:
        need_change = edge_checked.pop(cur_key)
        print("edge ", need_change.key, " from checked", need_change.joints)
    else:
        need_change = not_checked.pop(cur_key)
        print("edge ", need_change.key, " from Unchecked", need_change.joints)
        
    if need_change.joints[0] == need_replace:
        need_change.joints[0] = update_to
    else:
        if len(need_change.joints) > 1 and need_change.joints[1] == need_replace:
            need_change.joints[1] = update_to
        else:
            print(need_replace, " need to be ", update_to)
            print("error find:", need_change.key, need_change.joints)
        
    not_checked[cur_key] = need_change
    
def doFlip2(EdgesDict):
    SortedEdges = collections.OrderedDict(sorted(EdgesDict.items()))
    # print (SortedEdges)  
    # for test in SortedEdges:
    #     # print(test.key, test.joints)
    #     print(test, SortedEdges[test], SortedEdges[test].key, SortedEdges[test].joints)
    
    flips = 0
    
    while bool(SortedEdges):
        illegal_tri = []
        cur_edge = SortedEdges.popitem(last=False)[1]
        # print(cur_edge, cur_edge.key, cur_edge.low, cur_edge.high, cur_edge.joints)
        
        if cur_edge.isLegal():
            continue
        
        print(cur_edge.key, cur_edge.joints, 'illegal, need flipping +1')
        illegal_tri.append([cur_edge.low, cur_edge.high, cur_edge.joints[0]])
        illegal_tri.append([cur_edge.low, cur_edge.high, cur_edge.joints[1]])
        
        drawtriangles(illegal_tri, debug_mode=True)
    
    
def doFlip(EdgesDict):
    SortedEdges = collections.OrderedDict(sorted(EdgesDict.items()))
    # print (SortedEdges)  
    for test in SortedEdges:
        # print(test.key, test.joints)
        print(test, SortedEdges[test], SortedEdges[test].key, SortedEdges[test].joints)
    
    flips = 0
    checked_edges = {}
    while bool(SortedEdges):
        cur_edge = SortedEdges.popitem(last=False)[1]
        print(cur_edge, cur_edge.key, cur_edge.low, cur_edge.high, cur_edge.joints)
        
        if cur_edge.isLegal():
            checked_edges[cur_edge.key] = cur_edge
            continue
        
        print(cur_edge.key, 'illegal, need flipping +1')
        flips += 1
        # need flip   
        cur_edge.flip()
        checked_edges[cur_edge.key] = cur_edge
        
        print("checkedEdges before:")
        for test in checked_edges:
            print(test, checked_edges[test], checked_edges[test].key, checked_edges[test].joints)
        print("SortedEdges before:")
        for test in SortedEdges:
            # print(test.key, test.joints)
            print(test, SortedEdges[test], SortedEdges[test].key, SortedEdges[test].joints)
        
        update_edge(cur_edge.low, cur_edge.joints[0], cur_edge.high, cur_edge.joints[1], checked_edges, SortedEdges)
        update_edge(cur_edge.low, cur_edge.joints[1], cur_edge.high, cur_edge.joints[0], checked_edges, SortedEdges)
        update_edge(cur_edge.high, cur_edge.joints[0], cur_edge.low, cur_edge.joints[1], checked_edges, SortedEdges)
        update_edge(cur_edge.high, cur_edge.joints[1], cur_edge.low, cur_edge.joints[0], checked_edges, SortedEdges)
        
        print("checkedEdges after:")
        for test in checked_edges:
            print(test, checked_edges[test], checked_edges[test].key, checked_edges[test].joints)
        print("SortedEdges after:")
        for test in SortedEdges:
            # print(test.key, test.joints)
            print(test, SortedEdges[test], SortedEdges[test].key, SortedEdges[test].joints)
        # resort
        SortedEdges = collections.OrderedDict(sorted(SortedEdges.items()))
        
    return flips
    

def main():
    triangles = readTriangle("inputTriangulation5.txt")
    print(POINTS_INDEX)
    print(triangles)
    # easy view
    drawtriangles(triangles)
    
    #collect edge
    EdgesDict = {}
    
    for tri in triangles:
        collectEdgesFromTriangle(tri, EdgesDict)
    
    print (EdgesDict.keys())
    print ("flip times:", doFlip(EdgesDict))
    
    # return EdgesDict
    
main()