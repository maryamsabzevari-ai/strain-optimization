#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:40:20 2020

@author: maryam
"""
import numpy as np
from smt.sampling_methods import LHS
#import matplotlib.pyplot as plt

np.random.seed(1)

x = np.array([0.82,-1])
num = 100
dim=7
par=50
limits_dim = np.tile(x, (dim, 1))
LHS_data=np.zeros((par,num,dim))
for s in range(par):
    sampling = LHS(xlimits=limits_dim)
    LHS_data[s,] = sampling(num)
np.save("./dim"+str(dim)+"_LHS", LHS_data)
#######################################################################################################
dim=13
limits_dim = np.tile(x, (dim, 1))
LHS_data=np.zeros((par,num,dim))
for s in range(par):
    sampling = LHS(xlimits=limits_dim)
    LHS_data[s,] = sampling(num)
np.save("./dim"+str(dim)+"_LHS", LHS_data)
######################################################################################################
dim=6
limits_dim = np.tile(x, (dim, 1))
LHS_data=np.zeros((par,num,dim))
for s in range(par):
    sampling = LHS(xlimits=limits_dim)
    LHS_data[s,] = sampling(num)
np.save("./dim"+str(dim)+"_LHS", LHS_data)

