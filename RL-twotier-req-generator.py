#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, getopt, pdb
import time

import numpy as np
import pandas as pd
import random

pd.set_option("max_rows", 30)

import warnings

sys.path.append('./Class_func/')
from env import env
from req_generator import Req_generator_twotier

##load data
tier1=pd.read_csv('./info_table/table_fast_tier.csv')
tier1['time_untouched']=0
tier2=pd.read_csv('./info_table/table_slow_tier.csv')
tier2['time_untouched']=0

env1=env(tier1)
env2=env(tier2)

req_table=pd.DataFrame(columns=['No.','request','response'])
turn=0

while len(req_table)<50000:
    turn+=1
    print('%dth turn'%turn)
    ## generate requests
    Request=Req_generator_twotier(env1.tier,env2.tier)
    req_table=pd.concat([req_table,Request],ignore_index=True)
    req_table=req_table.loc[req_table['request']==1].reset_index(drop=True)

req_table=req_table[0:50000]
req_table.to_csv('requests_twotier.csv',index=False)
print('Complete!')
