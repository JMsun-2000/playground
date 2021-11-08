#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:31:51 2021

@author: sunjim
"""

from sortedcontainers import SortedList
from fractions import Fraction
from functools import total_ordering

import numpy as np
import pylab as pl
from matplotlib import collections  as mc






class Event:
    class Type:
        INTERSECTION = 0
        START = 1
        END = 2
    
    def __init__(self, type, segment, segment2 = None, ipoint = None):
        self.type = type
        self.segment = segment
        self.segment2 = segment2 # segment2 = None for non-intersection events
        
        '''
        此处用 -x做key是为了方便排序， 取反后，左边的点-x 大于 右边-x
        '''
        if type == 1:
            if segment.y1 < segment.y2 or (segment.y1 == segment.y2 and segment.x2 < segment.x1): # top-left as start key
                self.key = (segment.y2, -segment.x2)
            else:
                self.key = (segment.y1, -segment.x1)
        elif type == 2:
            if segment.y1 > segment.y2 or (segment.y1 == segment.y2 and segment.x2 > segment.x1): # bottom-right as end key
                self.key = (segment.y2, -segment.x2)
            else:
                self.key = (segment.y1, -segment.x1)
        elif type == 0:
            self.key = (ipoint[1], -ipoint[0]) # intersection key is (y, -x)

class Segment:
    def __init__(self, x1, y1, x2, y2, currentYList=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.currentYList = currentYList # stored in list to have it mutable
        
        if y1 == y2: 
            raise Exception('Horizontal segments not supported.')
        if x1 == x2: 
            raise Exception('Vertical segments not supported.')
        else:
            self.slope = (y2-y1)/(x2-x1)
            self.intersept = y2 - (self.slope * x2)   # y轴的截距
    def currentX(self):
        return (self.currentYList[0] - self.intersept) / self.slope  #任意Y，得到此线上的X
    
    #TODO: when inserting into the status we need a comparison operator between segments. t
    # That is we need to implement "<" (that is __lt__) and "==" (that is __eq__). 
    # What attribute(s) or method(s) do we need to call?
    # Below I filled in x1 (self.x1 < other.x1; self.x1 == other.x1), but that is certainly wrong
    # Change these two lines to make the comparison work correctly
    def __lt__(self, other):
        #return self.x1 < other.x1
        return self.currentX() < other.currentX()
    
    def __eq__(self, other):
        return self.currentX() == other.currentX()
    
    def intersection(self, other):
        x1 = self.x1
        y1 = self.y1
        x2 = self.x2
        y2 = self.y2
        x = other.x1
        y = other.y1
        xB = other.x2
        yB = other.y2
        
        dx1 = x2 - x1;  dy1 = y2 - y1
        dx = xB - x;  dy = yB - y;
        DET = (-dx1 * dy + dy1 * dx)
        
        #if math.fabs(DET) < DET_TOLERANCE: raise Exception('...')
        if DET == 0: raise Exception('Intersection implementation not sufficiently robust for this input.')
        DETinv = Fraction(1, DET)
                                     
        r = Fraction((-dy  * (x-x1) +  dx * (y-y1)), DET)
        s = Fraction((-dy1 * (x-x1) + dx1 * (y-y1)), DET)

        if r<0 or r>1 or s<0 or s>1: return None
        
        # return the average of the two descriptions
        xi = x1 + r*dx1
        yi = y1 + r*dy1
        return (xi, yi)

def checkIntersection(pos, pos2, Events, Status):
    segment = Status[pos]
    segment2 = Status[pos2]
    ipoint = segment.intersection(segment2)
    if ipoint and ipoint[1] < segment.currentYList[0]:
        ievent = Event(0, segment, segment2, ipoint)
        # Return an index to insert value in the sorted list, If the value is already present, the insertion point will be before (to the left of) any existing values.
        index = Events.bisect_left(ievent)
        # check if existing.
        if index == len(Events) or not Events[index].key == (ipoint[1], -ipoint[0]):
            Events.add(ievent)

def handleStartEvent(segment, Events, Status):
   # print("StartEvent Before", len(Status))
    Status.add(segment)
    # print("End", len(Status))
    # print("Status:", segment)
    # for s in Status: print(s, s.x1, s.y1, s.x2, s.y2, s.currentX())
    
    pos = Status.index(segment)
    if pos > 0: checkIntersection(pos, pos-1, Events, Status)
    if pos + 1 < len(Status) : checkIntersection(pos, pos+1, Events, Status)
    
def handelEndEvent(segment, Events, Status):
    # print("EndEvent Status:", segment)
    pos = Status.index(segment)
    # print("Before", len(Status))
    Status.remove(segment)
    # print("End", len(Status))
    # for s in Status: print(s, s.x1, s.y1, s.x2, s.y2, s.currentX())
    if pos > 0 and pos < len(Status):
        checkIntersection(pos, pos-1, Events, Status)
        #for s in Status: print("endevent, checkint", s, s.x1, s.y1, s.x2, s.y2, s.currentX())

def handleIntersectionEvent(segment, segment2, Events, Status):
    currentY = segment.currentYList[0] # to handle 2 segments have same Y, which may be hard to assure the order.
    segment.currentYList[0] = currentY + 0.00001 # we need to make sure that the comparison operator is consistent with the order just before the event
    
    #print("Handling intersection", segment.x1, segment.y1, segment.x2, segment.y2, segment2.x1, segment2.y1, segment2.x2, segment2.y2)
    
    ## swapping is not implemented for SortedList, so we need a trick here
    pos = Status.index(segment)
    pos2 = Status.index(segment2)
    ## instead we can remove and reinsert one of the segments. 
    Status.remove(segment)
    segment.currentYList[0] = currentY - 0.00001
    Status.add(segment)
    
    pos_first = min(pos, pos2)
    #交叉后, 调换位置的线性，左边线再和自己的左边，右边再和自己的右边，分别判断是否有交叉点
    if pos_first > 0: checkIntersection(pos_first-1, pos_first ,Events, Status)
    if pos_first + 2 < len(Status): checkIntersection(pos_first+1, pos_first+2, Events, Status) 
        
    segment.currentYList[0] = currentY # set back
    
    
def sweep(inputSegments):
    EndpointEvents = []
    for s in inputSegments:
        EndpointEvents.append(Event(1, s)) # segment start event
        EndpointEvents.append(Event(2, s)) # segment end event
    
    Events = SortedList(EndpointEvents, key=lambda x: x.key)
    Status = SortedList()
    nEvents = 0
    nIntersections = 0
    
    currentYList = inputSegments[0].currentYList   # currentYList = [0]
    lines = []
    
    while Events:
        e= Events.pop()
        nEvents += 1
        currentYList[0] = e.key[0] # y coordinate
        
        #print(e.type, e.segment.x1, e.segment.y1,  e.segment.x2, e.segment.y2, e.segment.currentX(), e.segment.currentYList[0])
        
        if nEvents in [3, 17, 99]: # just for homework assignment
            print("event", nEvents, ":", e.type)
        
        if e.type == 1: #start point
            handleStartEvent(e.segment, Events, Status)
            lines.append([[e.segment.x1, e.segment.y1], [e.segment.x2, e.segment.y2]])
        elif e.type == 2:
            handelEndEvent(e.segment, Events, Status)
        else:
            nIntersections += 1
            handleIntersectionEvent(e.segment, e.segment2, Events, Status)
            
        # print("Status:")
        # for s in Status: print(s.x1, s.y1, s.x2, s.y2, s.currentX())
    print("Number of events:", nEvents)
    print("Number of intersections:", nIntersections)

def readSegments(seg_file):
    currentYList = [0]
    segments = []
    with open(seg_file) as f:
        for line in f:
            coord = [int(x) for x in line.split()]
            s = Segment(coord[0], coord[1], coord[2], coord[3], currentYList)
            segments.append(s)
    return segments

seg_input = readSegments("LineSegments2.txt")
sweep(seg_input)