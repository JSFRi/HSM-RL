#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, getopt, pdb

import numpy as np
import pandas as pd
import random

##request generator
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

## Requests are generated from Poisson arrival process(lam=0.5 for hot file, 0.01 for cold file)
#@numba.jit(nopython=True)
def Req_generator(tier1,tier2,tier3):
    tier=pd.concat([tier1,tier2,tier3],ignore_index=True)
    No_list=[]
    req_list=[]
    for i in tier.index:
        No_list.append(tier['No.'][i])
        if tier['temp'][i]>0.5:
            req_list.append((1 if np.random.poisson(lam=0.5)>0 else 0))
        else:
            req_list.append((1 if np.random.poisson(lam=0.01)>0 else 0))
    
    ## simulate response time according to file size
    #resp=normalization(tier['weight'])
    resp=tier['weight']/10000
    
    return pd.DataFrame({'No.':No_list,'request':req_list,'response':resp})
    

def Req_generator_twotier(tier1,tier2):
    tier=pd.concat([tier1,tier2],ignore_index=True)
    No_list=[]
    req_list=[]
    for i in tier.index:
        No_list.append(tier['No.'][i])
        if tier['temp'][i]>0.5:
            req_list.append((1 if np.random.poisson(lam=0.5)>0 else 0))
        else:
            req_list.append((1 if np.random.poisson(lam=0.01)>0 else 0))
    
    ## simulate response time according to file size
    #resp=normalization(tier['weight'])
    resp=tier['weight']/1000000
    
    return pd.DataFrame({'No.':No_list,'request':req_list,'response':resp})

