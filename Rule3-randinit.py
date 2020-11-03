#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, getopt, pdb
import time

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
#from termcolor import colored, cprint

pd.set_option("max_rows", 30)

import warnings

#from IPython.display import clear_output # Used to clear the ouput of a Jupyter cell.

#from progressbar import *

#import numba

#import functions from other .py
from env import env, hot_cold, temp_decrease
from agent import TDAgent
from req_generator import Req_generator

def hot_cold_rule3(tier,file):
    file['req_time']+=1
    if float(file['weight']/file['req_time'])<200:
        temp=random.choice(np.arange(0.6,1.01,0.1))
        file['temp']=temp
    tier.loc[tier['No.']==int(file['No.'])]=file

##load data
all_tier=pd.read_csv('1000files.csv')
#all_tier
##
all_tier_rule3=all_tier
#all_tier_rule3['temp']=0
all_tier_rule3['req_time']=0

tier1=all_tier_rule3
tier2=all_tier_rule3[0:0]
tier3=all_tier_rule3[0:0]

transfer_list_rule3=[]

t0=time.time()

warnings.filterwarnings('ignore')
#progress=ProgressBar()
if os.path.exists('./Heatmap_rule3_tempdecrease_10000/')==False:
    os.mkdir('./Heatmap_rule3_tempdecrease_10000/')

env1=env(tier1)
env2=env(tier2)
env3=env(tier3)

for turn in range(10000):
    print('%dth turn'%turn)
    t1=time.time()
    ## generate requests
    #Request=Req_generator(env1.tier,env2.tier,env3.tier)
    ## use same requests
    Request=pd.read_csv('./Requests_randominit_10000/req_%d.csv'%turn)

    t2=time.time()
    print('Number of requests:',len(Request.loc[Request['request']==1]['No.']))
    
    ## count file transition numbers and volumes:
    num_t1_t3,num_t2_t3,num_t3_t2,num_t2_t1=[0]*4
    size_t1_t3,size_t2_t3,size_t3_t2,size_t2_t1=[0]*4
    
    ## begin file migration
    for req in list(Request.loc[Request['request']==1]['No.']):
        if req in list(env3.tier['No.']):
            ## if already in fastest tier, no movement
            ## file temperature change
            file=env3.tier.loc[env3.tier['No.']==req]
            hot_cold_rule3(env3.tier,file)
            
        elif req in list(env2.tier['No.']):
            ## judge if need to transit to tier3
             # get file infos
            file = env2.tier.loc[env2.tier['No.']==req]
            hot_cold_rule3(env2.tier,file)
            ## if hot then migrate, else don't
            file = env2.tier.loc[env2.tier['No.']==req]
            if float(file['temp'])>0.5:
                ## count transition number and volume:
                num_t2_t3+=1
                size_t2_t3+=float(file['weight'])
                ##
                env3.update_or_not(file, True)
                env2.update_or_not(file, True)
            else:
                pass
        else:
            ## judge if need to transit to tier2
             # get file infos
            file = env1.tier.loc[env1.tier['No.']==req]
            hot_cold_rule3(env1.tier,file)
            ## if hot then migrate, else don't
            file = env1.tier.loc[env1.tier['No.']==req]
            if float(file['temp'])>0.5:
                ## count transition number and volume:
                num_t1_t3+=1
                size_t1_t3+=float(file['weight'])
                ##
                env2.update_or_not(file, True)
                env1.update_or_not(file, True)
            else:
                pass
    print('Request loop time:',time.time()-t2,'seconds')
    
    ## check if any tier is out of space
    while env3.tier['weight'].sum()>100000:
        lowest_temp=env3.tier['temp'].min()
        Nos=list(env3.tier.loc[env3.tier['temp']==lowest_temp]['No.'])
        drop_No=random.choice(Nos)
        file=env3.tier.loc[env3.tier['No.']==drop_No]
        ## count transition number and volume:
        num_t3_t2+=1
        size_t3_t2+=float(file['weight'])
        ##
        env2.add_file(file)
        env3.remove_file(file)

    while env2.tier['weight'].sum()>1000000:
        lowest_temp=env2.tier['temp'].min()
        Nos=list(env2.tier.loc[env2.tier['temp']==lowest_temp]['No.'])
        drop_No=random.choice(Nos)
        file=env2.tier.loc[env2.tier['No.']==drop_No]
        ## count transition number and volume:
        num_t2_t1+=1
        size_t2_t1+=float(file['weight'])
        ##
        env1.add_file(file)
        env2.remove_file(file)
    
    print(num_t1_t3,'files being upgraded from Tier1 to Tier3, total sizes:',size_t1_t3)
    print(num_t2_t3,'files being upgraded from Tier2 to Tier3, total sizes:',size_t2_t3)
    print(num_t3_t2,'files being downgraded from Tier3 to Tier2, total sizes:',size_t3_t2)
    print(num_t2_t1,'files being downgraded from Tier2 to Tier1, total sizes:',size_t2_t1)
    transfer_list_rule3.append([num_t1_t3,size_t1_t3,num_t2_t3,size_t2_t3,num_t3_t2,size_t3_t2,num_t2_t1,size_t2_t1])
        
    ##Naturally decreasement of temperatures
    temp_decrease(env1.tier,env2.tier,env3.tier,Request,timestep=10)
    
    print('Process loop time:',time.time()-t1,'seconds')
        
    ## Draw heatmap for each tier
    heat1=list(env1.tier['temp'])+[0]*(1024-len(env1.tier['temp']))
    matrix_heat1=[]
    for i in range(len(heat1)//32):
        matrix_heat1.append(heat1[32*i:(32*i+32)])
    
    heat2=list(env2.tier['temp'])+[0]*(225-len(env2.tier['temp']))
    matrix_heat2=[]
    for i in range(len(heat2)//15):
        matrix_heat2.append(heat2[15*i:(15*i+15)])
        
    heat3=list(env3.tier['temp'])+[0]*(36-len(env3.tier['temp']))
    matrix_heat3=[]
    for i in range(len(heat3)//6):
        matrix_heat3.append(heat3[6*i:(6*i+6)])
    
    #clear_output(wait=True)
    
    plt.figure(1,figsize=(12,4), dpi=320)
    plt.subplot(1,3,1)
    plt.title('Tier1')
    sns.heatmap(matrix_heat1,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="YlGnBu")
    plt.subplot(1,3,2)
    plt.title('Tier2')
    sns.heatmap(matrix_heat2,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="YlGnBu")
    plt.subplot(1,3,3)
    plt.title('Tier3')
    sns.heatmap(matrix_heat3,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="YlGnBu")
    plt.savefig('./Heatmap_rule3_tempdecrease_10000/heatmap_%d.png'%turn,format='png',dpi=320)
    #plt.show()
    plt.close()
    
    #time.sleep(.1)
np.save('transfer_list_rule3_tempdecrease_10000.npy',np.array(transfer_list_rule3))
