# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:29:38 2021

@author: hillo
"""
import time
from direct_keys import PressKey, ReleaseKey, W, A, D, SPACE, R
for i in list(range(4)) [::-1]:
    print(i+1)
    time.sleep(1)
    
# print('down')
# PressKey(W)
# # time.sleep(3)
# print('up')
# ReleaseKey(W)

def forward():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def turn_left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)    
    
def turn_right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    
def stop():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    
def shoot():
    PressKey(SPACE)
    ReleaseKey(SPACE)
    
def restart():
    PressKey(R)
    ReleaseKey(R)