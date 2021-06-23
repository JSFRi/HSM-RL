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
from env import env, temp_decrease
from agent import TDAgent
from req_generator import Req_generator
##hot_cold related with file size & requests
## Temp switch func for file being requested
#@numba.jit(nopython=True)
def hot_cold(tier,file):
    tier.loc[tier['No.']==int(file['No.']),'req_time']+=1
    file=tier.loc[tier['No.']==int(file['No.'])]
    ## weight/request_time as probability ratio
    ratio=1-1/(1+np.exp(5000*7.33/10000)*np.exp(-(7.33/10000)*float(file['weight']/file['req_time'])))
    
    ##if file is cold, change it to hot according to the weight/req_time ratio
    if float(file['temp'])<0.5:
        if random.random()<0.5*ratio:
            temp=1-0.5/np.exp(0.01*file['req_time'])
            file['temp']=temp
            tier.loc[tier['No.']==int(file['No.'])]=file
        else:
            pass
     ##if the file is hot, increase temperatue by requests time
    else:
        temp=1-0.5/np.exp(0.01*file['req_time'])
        file['temp']=temp
        tier.loc[tier['No.']==int(file['No.'])]=file

#load tier tables
all_tier=pd.read_csv('1000files.csv')
all_tier['req_time']=0
all_tier['temp']=random.choices(np.arange(0.4,0.65,0.05),k=len(all_tier))
tier1=all_tier
tier2=all_tier[0:0]
tier3=all_tier[0:0]

## Initail parameters
b1t1=7.33/0.36
a1t1=np.exp(b1t1*0.35)
b2t1=7.33/1800
a2t1=np.exp(b2t1*1800)
b3t1=7.33/72
a3t1=np.exp(b3t1*60)

b1t2=7.33/0.54
a1t2=np.exp(b1t2*0.7)
b2t2=7.33/2500
a2t2=np.exp(b2t2*3000)
b3t2=7.33/42
a3t2=np.exp(b3t2*28)

b1t3=7.33/0.5
a1t3=np.exp(b1t3*0.75)
b2t3=7.33/4500
a2t3=np.exp(b2t3*4000)
b3t3=7.33/6.7
a3t3=np.exp(b3t3*3.3)

phi_t1_list=[[0]*8]
phi_t2_list=[[0]*8]
phi_t3_list=[[0]*8]

s1t1_list=[]
s2t1_list=[]
s3t1_list=[]
s1t2_list=[]
s2t2_list=[]
s3t2_list=[]
s1t3_list=[]
s2t3_list=[]
s3t3_list=[]

#transfer list
transfer_list_RL=[]
# repsonse time
response_list=[]
# request details
request_list=[]
# tier infos
tier_list=[]

t0=time.time()

warnings.filterwarnings('ignore')
#progress=ProgressBar()
if os.path.exists('./Heatmap_RL_zeroinit123-1000-temp46-fixnumreq-tempdesc5-0.01/')==False:
    os.mkdir('./Heatmap_RL_zeroinit123-1000-temp46-fixnumreq-tempdesc5-0.01/')

env1=env(tier1)
env2=env(tier2)
env3=env(tier3)

agent1=TDAgent(n_states=3,p_init=[0.0]*8,beta=0.05,lam=0.8,a_i=[a1t1,a2t1,a3t1],b_i=[b1t1,b2t1,b3t1])
agent2=TDAgent(n_states=3,p_init=[0.0]*8,beta=0.05,lam=0.8,a_i=[a1t1,a2t1,a3t1],b_i=[b1t1,b2t1,b3t1])
agent3=TDAgent(n_states=3,p_init=[0.0]*8,beta=0.05,lam=0.8,a_i=[a1t1,a2t1,a3t1],b_i=[b1t1,b2t1,b3t1])

for turn in range(1000):
    print('%dth turn'%turn)
    t1=time.time()
    ## generate requests
    Request=Req_generator(env1.tier,env2.tier,env3.tier)
    Request=Request.loc[Request['request']==1]
    req_limit=random.randint(170,230)
    Request=Request.sample(frac=1)[0:req_limit]
    #req_list.append(Request)
    
    #Request=pd.read_csv('./Requests_randominit_10000/req_%d.csv'%turn)
    
    ## calculate response time
    response=0
    for req in list(Request.loc[Request['request']==1]['No.']):
        if req in list(env3.tier['No.']):
            ### if in fastest tier, short response time
            response+=float(env3.tier.loc[env3.tier['No.']==req]['weight']/10000)
        elif req in list(env2.tier['No.']):
            ### if in medium tier, medium response time
            response+=float(10*env2.tier.loc[env2.tier['No.']==req]['weight']/10000)
        else:
            ### if in slowest tier, long response time
            response+=float(100*env1.tier.loc[env1.tier['No.']==req]['weight']/10000)
    response_list.append(response)
    
    ## record the request infos
    request_weight=0
    request_large=0
            
    ## calculate s1,s2,s3(n) & rewards
    s1t1_not,s2t1_not,s3t1_not,reward_t1=env1.step(Request)
    s1t2_not,s2t2_not,s3t2_not,reward_t2=env2.step(Request)
    s1t3_not,s2t3_not,s3t3_not,reward_t3=env3.step(Request)
    
    ## Record s1,s2,s3
    s1t1_list.append(s1t1_not)
    s2t1_list.append(s2t1_not)
    s3t1_list.append(s3t1_not)
    s1t2_list.append(s1t2_not)
    s2t2_list.append(s2t2_not)
    s3t2_list.append(s3t2_not)
    s1t3_list.append(s1t3_not)
    s2t3_list.append(s2t3_not)
    s3t3_list.append(s3t3_not)

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
            if float(file['weight'])>5000:
                request_large+=1
            
        elif req in list(env2.tier['No.']):
            ## judge if need to transit to tier3
             # get file infos
            file = env2.tier.loc[env2.tier['No.']==req]
            hot_cold(env2.tier,file)
            file = env2.tier.loc[env2.tier['No.']==req]
            
            # record req details
            request_weight+=float(file['weight'])
            if float(file['weight'])>5000:
                request_large+=1
                
             # calculate c_up&c_not
            C_not_t3, s1_not_t3, C_up_t3, s1_up_t3=agent3.c_up_c_not(env3.tier,file,Request)
            C_not_t2, s1_not_t2, C_up_t2, s1_up_t2=agent2.c_up_c_not(env2.tier,file,Request)
             # criterion
            if C_up_t3*s1_up_t3+C_up_t2*s1_up_t2 > C_not_t3*s1_not_t3+C_not_t2*s1_not_t2:
                # then upgrade
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
            file = env1.tier.loc[env1.tier['No.']==req]
            
            # record req details
            request_weight+=float(file['weight'])
            if float(file['weight'])>5000:
                request_large+=1
                
             # calculate c_up&c_not
            C_not_t3, s1_not_t3, C_up_t3, s1_up_t3=agent2.c_up_c_not(env2.tier,file,Request)
            C_not_t1, s1_not_t1, C_up_t1, s1_up_t1=agent1.c_up_c_not(env1.tier,file,Request)
             # criterion
            if C_up_t1*s1_up_t1+C_up_t3*s1_up_t3 > C_not_t1*s1_not_t1+C_not_t3*s1_not_t3:
                # then upgrade
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
    transfer_list_RL.append([num_t1_t3,size_t1_t3,num_t2_t3,size_t2_t3,num_t3_t2,size_t3_t2,num_t2_t1,size_t2_t1])
    
    request_list.append([len(Request.loc[Request['request']==1]['No.']),request_weight,request_large])
    
    tier_list.append([env1.tier.to_numpy(),env2.tier.to_numpy(),env3.tier.to_numpy()])

    
    ## calculate new s1,s2,s3 
    s1t1_up,s2t1_up,s3t1_up,_=env1.step(Request)
    s1t2_up,s2t2_up,s3t2_up,_=env2.step(Request)
    s1t3_up,s2t3_up,s3t3_up,_=env3.step(Request)
    
    ## agent learn
    phi_t1=agent1.learn([s1t1_not,s2t1_not,s3t1_not], reward_t1, [s1t1_up,s2t1_up,s3t1_up], phi_t1_list)
    phi_t1_list.append(phi_t1)
    
    phi_t2=agent2.learn([s1t2_not,s2t2_not,s3t2_not], reward_t2, [s1t2_up,s2t2_up,s3t2_up], phi_t2_list)
    phi_t2_list.append(phi_t2)
    
    phi_t3=agent3.learn([s1t3_not,s2t3_not,s3t3_not], reward_t3, [s1t3_up,s2t3_up,s3t3_up], phi_t3_list)
    phi_t3_list.append(phi_t3)
    
    ##Naturally decreasement of temperatures
    temp_decrease(env1.tier,env2.tier,env3.tier,Request,timestep=5)
    
    ## Draw heatmap for each tier
    heat1=list(env1.tier['temp'])+[0]*(1024-len(env1.tier['temp']))
    matrix_heat1=[]
    for i in range(len(heat1)//32):
        matrix_heat1.append(heat1[32*i:(32*i+32)])
    
    heat2=list(env2.tier['temp'])+[0]*(1024-len(env2.tier['temp']))
    matrix_heat2=[]
    for i in range(len(heat2)//32):
        matrix_heat2.append(heat2[32*i:(32*i+32)])
        
    heat3=list(env3.tier['temp'])+[0]*(1024-len(env3.tier['temp']))
    matrix_heat3=[]
    for i in range(len(heat3)//32):
        matrix_heat3.append(heat3[32*i:(32*i+32)])
    
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
    plt.savefig('./Heatmap_RL_zeroinit123-1000-temp46-fixnumreq-tempdesc5-0.01/heatmap_%d.png'%turn,format='png',dpi=320)
    #plt.show()
    plt.close()
    
    '''## Trace largest/smallest & hottest/coldest file
    if 70 in list(env1.tier['No.']):
        print('Large hot file in Tier1')
    elif 70 in list(env2.tier['No.']):
        print('Large Hot file in Tier2')
    else:
        print('Large Hot file in Tier3')
        
    if 21 in list(env1.tier['No.']):
        print('Large cold file in Tier1')
    elif 21 in list(env2.tier['No.']):
        print('Large cold file in Tier2')
    else:
        print('Large cold file in Tier3')
        
    if 177 in list(env1.tier['No.']):
        print('Small hot file in Tier1')
    elif 177 in list(env2.tier['No.']):
        print('Small Hot file in Tier2')
    else:
        print('Small Hot file in Tier3')
        
    if 476 in list(env1.tier['No.']):
        print('Small cold file in Tier1')
    elif 476 in list(env2.tier['No.']):
        print('Small cold file in Tier2')
    else:
        print('Small cold file in Tier3')'''
    
    #time.sleep(.1)
    print('Total time for one turn:',time.time()-t1,'seconds')
print('Total time:',time.time()-t0,'seconds')
np.save('transfer_list_RL_1000_fixnumrandpoireq_zeroinit123_temp46_tempdesc5_0.01.npy',np.array(transfer_list_RL))
np.save('response_list_RL_1000_fixnumrandpoireq_zeroinit123_temp46_tempdesc5_0.01.npy',np.array(response_list))
np.save('request_list_RL_1000_fixnumrandpoireq_zeroinit123_temp46_tempdesc5_0.01.npy',np.array(request_list))
np.save('Tier_list_RL_1000_fixnumrandpoireq_zeroinit123_temp46_tempdesc5_0.01.npy',np.array(tier_list))