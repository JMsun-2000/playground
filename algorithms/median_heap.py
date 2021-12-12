#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 20:07:27 2021

@author: sunjim
"""

def readNumber(number_file):
    number_array = []
    
    with open(number_file) as f:
        for line in f:
            element = [x for x in line.split()][0]
            number_array.append(int(element))
    return number_array


class Heap:
    def __init__(self, low_top=True):
        self.heapspace = []
        self.low_top = low_top
        
    def length(self):
        return len(self.heapspace)
    
    def topValue(self):
        if self.length() == 0:
            return None
        else:
            return self.heapspace[0]
    
    def popTop(self):
        if self.length() == 0:
            return None
        
        extracted = self.heapspace[0]
        self.heapspace[0] = self.heapspace.pop()
        cur_pos = 1
        
        while (cur_pos*2) <= self.length():
            left_child = cur_pos * 2
            right_child = left_child + 1
            
            if self.low_top:
                if self.heapspace[cur_pos-1] <= self.heapspace[left_child-1] and (right_child > self.length() or self.heapspace[cur_pos-1] <= self.heapspace[right_child-1]):
                    break
            else:
                if self.heapspace[cur_pos-1] >= self.heapspace[left_child-1] and (right_child > self.length() or self.heapspace[cur_pos-1] >= self.heapspace[right_child-1]):
                    break
            
            # need switch
            switch_pos = cur_pos
            if right_child > self.length():
                switch_pos = left_child
            else:                  
                if self.low_top:
                    switch_pos = left_child if self.heapspace[left_child-1] <= self.heapspace[right_child-1] else right_child
                else:
                    switch_pos = left_child if self.heapspace[left_child-1] >= self.heapspace[right_child-1] else right_child
            
            self.heapspace[cur_pos-1], self.heapspace[switch_pos-1] = self.heapspace[switch_pos-1], self.heapspace[cur_pos-1]
            cur_pos = switch_pos
        
        
        return extracted
    
    def insert(self, value):
       self.heapspace.append(value)
       new_position = self.length()
       while new_position > 1:
           parent_position = new_position//2
           if self.low_top:
               if self.heapspace[parent_position-1] <= self.heapspace[new_position-1]:
                   # perfect
                   break
           else:
               if self.heapspace[parent_position-1] >= self.heapspace[new_position-1]:
                   # perfect
                   break
             
           # switch
           self.heapspace[parent_position-1], self.heapspace[new_position-1] = self.heapspace[new_position-1], self.heapspace[parent_position-1]
           new_position = parent_position
           
       return new_position-1


   
def Main():
    number_input = readNumber("MedianClass2-3.txt")
    small_heap = Heap(False)  
    big_heap = Heap(True)
    median_sum = 0
    
    debug_count = 0
    
    for xi in number_input:
        debug_count +=1
        # insert
        if small_heap.topValue() == None:
            small_heap.insert(xi)
        else:
            if small_heap.topValue() > xi:
                small_heap.insert(xi)
            else:
                big_heap.insert(xi)  
                
        # if debug_count == 12:
        #    print ('round',debug_count,'insert', xi) 
        #    print ('small',small_heap.heapspace)
        #    print ('big',big_heap.heapspace)
        # balance
        if small_heap.length() - big_heap.length() > 1:
            big_heap.insert(small_heap.popTop())
        elif big_heap.length() - small_heap.length() > 1:
            small_heap.insert(big_heap.popTop())
        
        # if debug_count == 12:
        #     print ('small',small_heap.heapspace)
        #     print ('big',big_heap.heapspace)
        # if small_heap.length() + big_heap.length() > 20:
        #     break
            
        # get median
        if small_heap.length() >= big_heap.length():
            median_sum += small_heap.topValue()
            # print ('median', small_heap.topValue())
        else:
            median_sum += big_heap.topValue()
           # print ('median', big_heap.topValue())
       # print('--------------------------')
            
    return median_sum

median_all = Main()
print ('median_sum',median_all)
print ('mod 10000', median_all%10000)
                
                
        

           






       
