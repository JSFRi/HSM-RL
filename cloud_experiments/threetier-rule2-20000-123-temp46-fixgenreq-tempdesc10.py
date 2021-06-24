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
sys.path.append('/home/ubuntu/HSM-RL/Class_func/')
from env import env, hot_cold, temp_decrease
from agent import TDAgent
from req_generator import Req_generator


#load tier tables
all_tier=pd.read_csv('20000files.csv')
all_tier['req_time']=0
all_tier['temp']=random.choices(np.arange(0.4,0.65,0.05),k=len(all_tier))
tier1=all_tier
tier2=all_tier[0:0]
tier3=all_tier[0:0]


#transfer list
transfer_list_rule2=[]
# repsonse time
response_list_rule2=[]
# request details
request_list=[]
# tier infos
tier_list=[]

t0=time.time()

warnings.filterwarnings('ignore')
#progress=ProgressBar()
if os.path.exists('./Heatmap_2ndrule_20000-123-temp46-fixgenreq-tempdesc10/')==False:
    os.mkdir('./Heatmap_2ndrule_20000-123-temp46-fixgenreq-tempdesc10/')

env1=env(tier1)
env2=env(tier2)
env3=env(tier3)

for turn in range(1000):
    print('%dth turn'%turn)
    t1=time.time()
    ## generate requests
    #Request=Req_generator(env1.tier,env2.tier,env3.tier)
    #Request=Request.loc[Request['request']==1]
    #req_limit=random.randint(170,230)
    #Request=Request.sample(frac=1)[0:req_limit]
    ## use same requests
    Request=pd.read_csv('./Requests_temp46_20000/req_%d.csv'%turn)

    ## calculate response time
    response=0
    for req in list(Request.loc[Request['request']==1]['No.']):
        if req in list(env3.tier['No.']):
            ### if in fastest tier, short response time
            response+=float(env3.tier.loc[env3.tier['No.']==req]['weight']/10000000)
        elif req in list(env2.tier['No.']):
            ### if in medium tier, medium response time
            response+=float(10*env2.tier.loc[env2.tier['No.']==req]['weight']/10000000)
        else:
            ### if in slowest tier, long response time
            response+=float(100*env1.tier.loc[env1.tier['No.']==req]['weight']/10000000)
    response_list_rule2.append(response)
    
    ## record the request infos
    request_weight=0
    request_large=0
    
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
            hot_cold(env3.tier,file)
            
            # record req details
            request_weight+=float(file['weight'])
            if float(file['weight'])>50000000:
                request_large+=1
            
        elif req in list(env2.tier['No.']):
            ## judge if need to transit to tier3
             # get file infos
            file = env2.tier.loc[env2.tier['No.']==req]
            hot_cold(env2.tier,file)
            ## if hot then migrate, else don't
            file = env2.tier.loc[env2.tier['No.']==req]
            
            # record req details
            request_weight+=float(file['weight'])
            if float(file['weight'])>50000000:
                request_large+=1
                
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
            ## judge if need to transit to tier3
             # get file infos
            file = env1.tier.loc[env1.tier['No.']==req]
            hot_cold(env1.tier,file)
            ## if hot then migrate, else don't
            file = env1.tier.loc[env1.tier['No.']==req]
            
            # record req details
            request_weight+=float(file['weight'])
            if float(file['weight'])>50000000:
                request_large+=1
                
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
    while env3.tier['weight'].sum()>2000000000:
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

    while env2.tier['weight'].sum()>6000000000:
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
        
    ##Naturally decreasement of temperatures
    temp_decrease(env1.tier,env2.tier,env3.tier,Request,timestep=10)
    
    print(num_t1_t3,'files being upgraded from Tier1 to Tier3, total sizes:',size_t1_t3)
    print(num_t2_t3,'files being upgraded from Tier2 to Tier3, total sizes:',size_t2_t3)
    print(num_t3_t2,'files being downgraded from Tier3 to Tier2, total sizes:',size_t3_t2)
    print(num_t2_t1,'files being downgraded from Tier2 to Tier1, total sizes:',size_t2_t1)
    transfer_list_rule2.append([num_t1_t3,size_t1_t3,num_t2_t3,size_t2_t3,num_t3_t2,size_t3_t2,num_t2_t1,size_t2_t1])
    
    request_list.append([len(Request.loc[Request['request']==1]['No.']),request_weight,request_large])
    
    tier_list.append([env1.tier.to_numpy(),env2.tier.to_numpy(),env3.tier.to_numpy()])
    
        
    ## Draw heatmap for each tier
    heat1=list(env1.tier['temp'])+[0]*(20164-len(env1.tier['temp']))
    matrix_heat1=[]
    for i in range(len(heat1)//142):
        matrix_heat1.append(heat1[142*i:(142*i+142)])
    
    heat2=list(env2.tier['temp'])+[0]*(20164-len(env2.tier['temp']))
    matrix_heat2=[]
    for i in range(len(heat2)//142):
        matrix_heat2.append(heat2[142*i:(142*i+142)])
        
    heat3=list(env3.tier['temp'])+[0]*(20164-len(env3.tier['temp']))
    matrix_heat3=[]
    for i in range(len(heat3)//142):
        matrix_heat3.append(heat3[142*i:(142*i+142)])
    
    #clear_output(wait=True)
    
    plt.figure(turn,figsize=(12,4), dpi=320)
    plt.subplot(1,3,1)
    plt.title('Tier1')
    sns.heatmap(matrix_heat1,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")
    plt.subplot(1,3,2)
    plt.title('Tier2')
    sns.heatmap(matrix_heat2,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")
    plt.subplot(1,3,3)
    plt.title('Tier3')
    sns.heatmap(matrix_heat3,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")
    plt.savefig('./Heatmap_2ndrule_20000-123-temp46-fixgenreq-tempdesc10/heatmap_%d.png'%turn,format='png',dpi=320)
    #plt.show()
    plt.close()
    
    #time.sleep(.1)
    print('Total time for one turn:',time.time()-t1,'seconds')
print('Total time:',time.time()-t0,'seconds')
np.save('transfer_list_rule2_20000_123_fixgenreq_temp46_tempdesc10.npy',np.array(transfer_list_rule2))
np.save('response_list_rule2_20000_123_fixgenreq_temp46_tempdesc10.npy',np.array(response_list_rule2))
np.save('request_list_rule2_20000_123_fixgenreq_temp46_tempdesc10.npy',np.array(request_list))
np.save('Tier_list_rule2_20000_123_fixgenreq_temp46_tempdesc10.npy',np.array(tier_list))