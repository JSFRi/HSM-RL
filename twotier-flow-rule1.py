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


#import functions from other .py
sys.path.append('./Class_func/')
from env import env, hot_cold, temp_decrease_twotier
from agent import TDAgent
from req_generator import Req_generator_twotier

#import celery worker
from worker1 import fast_move_file, fast_get_file
from worker2 import slow_move_file, slow_get_file

##load data
tier1=pd.read_csv('./info_table/table_fast_tier.csv')
tier1['time_untouched']=0
tier2=pd.read_csv('./info_table/table_slow_tier.csv')
tier2['time_untouched']=0
##Initially Store files by Rule-1
all_tier=pd.concat([tier1,tier2],ignore_index=True)
sum_weight=0
for i in range(len(all_tier)):
    weight=all_tier['weight'][i]
    sum_weight=sum_weight+weight
    if sum_weight>16000000: ##80% of tier1
        break
tier1=all_tier[0:i].reset_index(drop=True)
tier2=all_tier[i:].reset_index(drop=True)

##load requests
Requests=pd.read_csv('requests_twotier.csv')

## record variables


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
transfer_real_RL=[]

t0=time.time()

warnings.filterwarnings('ignore')
#progress=ProgressBar()
if os.path.exists('./Heatmap_RL_twotier_Rule1/')==False:
    os.mkdir('./Heatmap_RL_twotier_Rule1/')

env1=env(tier1)
env2=env(tier2)

count=0
t1=time.time()

num_per_turn=input('Please input number of requests as one turn: ')
num_per_turn=int(num_per_turn)

## %input requests as one turn
for turn in range(int(100000/num_per_turn)):
   
    ## generate requests
    Request=Requests[turn*num_per_turn:(turn*num_per_turn+num_per_turn)]
    
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
            ## if hot then migrate, else don't
            file = env2.tier.loc[env2.tier['No.']==req]
            if float(file['temp'])>0.5:
                ## count transition number and volume:
                num_t2_t1+=1
                size_t2_t1+=float(file['weight'])
                ##
                env1.update_or_not(file, True)
                env2.update_or_not(file, True)
            else:
                pass
    
    ## check if any tier is out of space
    while env1.tier['weight'].sum()>2000000000:
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
        
    ## Record transfer numbers    
    transfer_list_RL.append([num_t1_t2,num_t2_t1,size_t1_t2,size_t2_t1])

    ##Naturally decreasement of temperatures
    temp_decrease_twotier(env1.tier,env2.tier,Request,timestep=500)
    
    ##Settlement after each num_per_turn requests
    #if count%num_per_turn==0:
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
    plt.savefig('./Heatmap_RL_twotier_Rule1/heatmap_%d.png'%turn,format='png',dpi=320)
    #plt.show()
    plt.close()
    
    ##send file migration infos to fast/slow tier
    #gather file numbers that should be downgraded from t1 to t2
    no_t1_t2=[]
    for i in tier1['No.']:
        if i not in list(env1.tier['No.']):
            no_t1_t2.append(i)
    #push files into nfs
    #for i in no_t1_t2:
        #fast_move_file.delay(i)
    print('Downgrade %d files from fast_tier to slow_tier'%(len(no_t1_t2)))
    #gather files from nfs to slow_tier
    #slow_get_file.delay(0)
    tier1=env1.tier
    
    #gather file numbers that should be upgraded from t2 to t1
    no_t2_t1=[]
    for i in tier2['No.']:
        if i not in list(env2.tier['No.']):
            no_t2_t1.append(i)
    #push files into nfs
    #for i in no_t2_t1:
        #slow_move_file.delay(i)
    print('Upgrade %d files from slow_tier to fast_tier'%(len(no_t2_t1)))
    #gather files from nfs to fast_tier
    #fast_get_file.delay(0)
    tier2=env2.tier
    
    transfer_real_RL.append([len(no_t1_t2),len(no_t2_t1)])
        
    
    ## request_count +num_per_turn
    count+=num_per_turn
    print(count,'requests have been proceeded,','using',time.time()-t1,'seconds')
    t1=time.time()
    
np.save('transfer_list_rule1_twotier_100000_%d.npy'%num_per_turn,np.array(transfer_list_RL))
np.save('transfer_real_rule1_twotier_100000_%d.npy'%num_per_turn,np.array(transfer_real_RL))

pd.DataFrame({'s1t1':s1t1_list,
              's2t1':s2t1_list,
              's3t1':s3t1_list,
              's1t2':s1t2_list,
              's2t2':s2t2_list,
              's3t2':s3t2_list,}).to_csv('s123_RL_twotier_100000_rule1_%d.csv'%num_per_turn,index=False)
              
print('Complete!')
