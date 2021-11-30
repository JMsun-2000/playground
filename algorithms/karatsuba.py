#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:00:57 2021

@author: sunjim
"""


'''
input: 
A = 3141592653589793238462643383279502884197169399375105820974944592
B = 2718281828459045235360287471352662497757247093699959574966967627

answer:
8539734222673567065463550869546574495034888535765114961879601127067743044893204848617875072216249073013374895871952806582723184

'''
from math import ceil

def karatuba_product(num_str_1, num_str_2):
    num_length = max(len(num_str_1), len(num_str_2))
    # less than 2, return result
    if len(num_str_1) <= 2 or len(num_str_2) <=2:
        return int(num_str_1) * int(num_str_2)
    
    half_a = ceil(num_length/2)

    a = num_str_1[:-half_a]
    b = num_str_1[-half_a:]
    c = num_str_2[:-half_a]
    d = num_str_2[-half_a:]
  #  print ("a:", a, "b:", b, "c:", c, "d:", d)
    
    ac = karatuba_product(a, c)
    bd = karatuba_product(b, d)
   
    ab_m_cd = karatuba_product(str(int(a)+int(b)), str(int(c)+int(d)))
    if ((int(a)+int(b))*(int(c)+int(d))) != ab_m_cd:
        print("a+b", str(int(a)+int(b)), "c+d", str(int(c)+int(d)))
        print("should:", ((int(a)+int(b))*(int(c)+int(d))), "actual:", ab_m_cd)
    
    return ac*pow(10, half_a*2)+bd+(ab_m_cd - ac - bd)*pow(10,half_a)


A = '3141592653589793238462643383279502884197169399375105820974944592'
B = '2718281828459045235360287471352662497757247093699959574966967627'
#print(karatuba_product(A, B))   
print(karatuba_product(A, B) == 8539734222673567065463550869546574495034888535765114961879601127067743044893204848617875072216249073013374895871952806582723184) 