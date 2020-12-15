#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, getopt, pdb

import numpy as np
import pandas as pd

#import functions from other .py
sys.path.append('./Class_func/')
from req_generator import Req_generator_twotier

tier1=pd.read_csv('./info_table/table_fast_tier.csv')
tier2=pd.read_csv('./info_table/table_slow_tier.csv')

Requests=pd.DataFrame(columns=['No.','request','response'])

num=len(Requests)

while num < 100000:
    reqs=Req_generator_twotier(tier1,tier2)
    reqs=reqs.loc[reqs['request']==1]
    Requests=pd.concat([Requests,reqs],ignore_index=True)
    num=len(Requests)
    print('generated %d requests'%num)

Requests=Requests[0:100000]

Requests.to_csv('requests_twotier.csv',index=False)

print('Done')
