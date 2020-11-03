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
sys.path.append('./RL/')
from env import env, hot_cold, temp_decrease_twotier
from agent import TDAgent
from req_generator import Req_generator_twotier

##load data
tier1=pd.read_csv('./info_table/table_fast_tier.csv')
tier1['time_untouched']=0
tier2=pd.read_csv('./info_table/table_slow_tier.csv')
tier2['time_untouched']=0

## Initail parameters
b1t1=7.33/0.5
a1t1=np.exp(b1t1*0.5)
b2t1=7.33/1000000
a2t1=np.exp(b2t1*100000)
b3t1=7.33/50
a3t1=np.exp(b3t1*75)

b1t2=7.33/0.5
a1t2=np.exp(b1t2*0.5)
b2t2=7.33/1000000
a2t2=np.exp(b2t2*100000)
b3t2=7.33/40
a3t2=np.exp(b3t2*30)


phi1s_t1_list=[0]
phi2s_t1_list=[0]
phi1s_t2_list=[0]
phi2s_t2_list=[0]

s1t1_list=[]
s2t1_list=[]
s3t1_list=[]
s1t2_list=[]
s2t2_list=[]
s3t2_list=[]

transfer_list_RL=[]

t0=time.time()

warnings.filterwarnings('ignore')
#progress=ProgressBar()
if os.path.exists('./Heatmap_RL_twotier_1000/')==False:
    os.mkdir('./Heatmap_RL_twotier_1000/')

env1=env(tier1)
env2=env(tier2)

agent1=TDAgent(n_states=3,p1_init=0,p2_init=0,beta=0.05,lam=0.8,a1=a1t1,a2=a2t1,a3=a3t1,b1=b1t1,b2=b2t1,b3=b3t1)
agent2=TDAgent(n_states=3,p1_init=0,p2_init=0,beta=0.05,lam=0.8,a1=a1t2,a2=a2t2,a3=a3t2,b1=b1t2,b2=b2t2,b3=b3t2)

for turn in range(1000):
    print('%dth turn'%turn)
    t1=time.time()
    ## generate requests
    Request=Req_generator_twotier(env1.tier,env2.tier)
    #Request.to_csv('./Requests_randinit_10000/req_%d.csv'%turn,index=False)
    #Request=pd.read_csv('./Requests_randominit_10000/req_%d.csv'%turn)
    
    ## calculate s1,s2,s3(n) & rewards
    s1t1_not,s2t1_not,s3t1_not,reward_t1=env1.step(Request)
    s1t2_not,s2t2_not,s3t2_not,reward_t2=env2.step(Request)
    
    ## Record s1,s2,s3
    s1t1_list.append(s1t1_not)
    s2t1_list.append(s2t1_not)
    s3t1_list.append(s3t1_not)
    s1t2_list.append(s1t2_not)
    s2t2_list.append(s2t2_not)
    s3t2_list.append(s3t2_not)

    t2=time.time()
    print('Number of requests:',len(Request.loc[Request['request']==1]['No.']))
    
    ## count file transition numbers and volumes:
    num_t1_t2,num_t2_t1=[0]*2
    size_t1_t2,size_t2_t1=[0]*2
    
    ## begin file migration
    for req in list(Request.loc[Request['request']==1]['No.']):
        if req in list(env2.tier['No.']):
        ## judge if need to transit to tier1
             # get file infos
            file = env2.tier.loc[env2.tier['No.']==req]
            hot_cold(env2.tier,file)
             # calculate c_up&c_not
            C_not_t1, s1_not_t1, C_up_t1, s1_up_t1=agent1.c_up_c_not(env1.tier,file)
            C_not_t2, s1_not_t2, C_up_t2, s1_up_t2=agent2.c_up_c_not(env2.tier,file)
             # criterion
            if C_up_t1*s1_up_t1+C_up_t2*s1_up_t2 < C_not_t1*s1_not_t1+C_not_t2*s1_not_t2:
                # then upgrade
                ## count transition number and volume:
                num_t2_t1+=1
                size_t2_t1+=float(file['weight'])
                ##
                env1.update_or_not(file, True)
                env2.update_or_not(file, True)
            else:
                pass
    print('Request loop time:',time.time()-t2,'seconds')
    
    ## check if any tier is out of space
    while env1.tier['weight'].sum()>10000000:
        lowest_temp=env1.tier['temp'].min()
        Nos=list(env1.tier.loc[env1.tier['temp']==lowest_temp]['No.'])
        drop_No=random.choice(Nos)
        file=env1.tier.loc[env1.tier['No.']==drop_No]
        ## count transition number and volume:
        num_t1_t2+=1
        size_t1_t2+=float(file['weight'])
        ##
        env2.add_file(file)
        env1.remove_file(file)

    
    print(num_t1_t2,'files being upgraded from Tier1 to Tier2, total sizes:',size_t1_t2)
    print(num_t2_t1,'files being downgraded from Tier2 to Tier1, total sizes:',size_t2_t1)
    transfer_list_RL.append([num_t1_t2,size_t1_t2,num_t2_t1,size_t2_t1])

    
    ## calculate new s1,s2,s3 
    s1t1_up,s2t1_up,s3t1_up,_=env1.step(Request)
    s1t2_up,s2t2_up,s3t2_up,_=env2.step(Request)
    
    ## agent learn
    phi1s_t1,_=agent1.learn([s1t1_not,s2t1_not,s3t1_not], reward_t1, [s1t1_up,s2t1_up,s3t1_up], phi1s_t1_list, phi2s_t1_list)
    phi1s_t1_list.append(phi1s_t1)
    _,phi2s_t1=agent1.learn([s1t1_not,s2t1_not,s3t1_not], reward_t1, [s1t1_up,s2t1_up,s3t1_up], phi1s_t1_list, phi2s_t1_list)
    phi2s_t1_list.append(phi2s_t1)
    
    phi1s_t2,_=agent2.learn([s1t2_not,s2t2_not,s3t2_not], reward_t2, [s1t2_up,s2t2_up,s3t2_up], phi1s_t2_list, phi2s_t2_list)
    phi1s_t2_list.append(phi1s_t2)
    _,phi2s_t2=agent2.learn([s1t2_not,s2t2_not,s3t2_not], reward_t2, [s1t2_up,s2t2_up,s3t2_up], phi1s_t2_list, phi2s_t2_list)
    phi2s_t2_list.append(phi2s_t2)
    
    ##Naturally decreasement of temperatures
    temp_decrease_twotier(env1.tier,env2.tier,Request,timestep=10)
    
    ## Draw heatmap for each tier
    heat1=list(env1.tier['temp'])+[0]*(16-len(env1.tier['temp']))
    matrix_heat1=[]
    for i in range(len(heat1)//4):
        matrix_heat1.append(heat1[4*i:(4*i+4)])
    
    heat2=list(env2.tier['temp'])+[0]*(225-len(env2.tier['temp']))
    matrix_heat2=[]
    for i in range(len(heat2)//15):
        matrix_heat2.append(heat2[15*i:(15*i+15)])
        
    #clear_output(wait=True)
    
    plt.figure(turn,figsize=(10,4), dpi=320)
    plt.subplot(1,2,1)
    plt.title('Tier1')
    sns.heatmap(matrix_heat1,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="YlGnBu")
    plt.subplot(1,2,2)
    plt.title('Tier2')
    sns.heatmap(matrix_heat2,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="YlGnBu")
    plt.savefig('./Heatmap_RL_twotier_1000/heatmap_%d.png'%turn,format='png',dpi=320)
    #plt.show()
    plt.close()
    
    #time.sleep(.1)
    print('Total time for one turn:',time.time()-t1,'seconds')
print('Total time:',time.time()-t0,'seconds')
np.save('transfer_list_RL_twotier_1000.npy',np.array(transfer_list_RL))
print('Complete!')
