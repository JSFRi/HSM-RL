#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, getopt, pdb

import numpy as np
import pandas as pd
import random

##TD() agent 
class TDAgent():
    
    def __init__(self,n_states,p_init,beta,lam,a_i,b_i):
        self.n_states = n_states
        # parameters of cost-to-go func
        self.p = np.array(p_init)
        # parameters of FRB func
        self.a1,self.a2,self.a3 = a_i
        self.b1,self.b2,self.b3 = b_i
        # parameters of TD
        self.alpha= [0]*len(self.p)
        self.beta = beta
        self.lam  = lam
        # eligibility trace
        self.z = [0]*len(self.p)
        
    def act(self, state):
        # Random
        return np.random.choice(self.n_actions)
    
    def cost_phi(self,state):
        s1,s2,s3=state
        
        # Fuzzy Rule-Based function
        ## membership function /mu
        mu_L1=(1/(1+self.a1*np.exp(-self.b1*s1)))
        mu_S1=1-mu_L1
        mu_L2=(1/(1+self.a2*np.exp(-self.b2*s2)))
        mu_S2=1-mu_L2
        mu_L3=(1/(1+self.a3*np.exp(-self.b3*s3)))
        mu_S3=1-mu_L3
        ## weight w^i
        w=[]
        for i in [mu_S1,mu_L1]:
            for j in [mu_S2,mu_L2]:
                for k in [mu_S3,mu_L3]:
                    w.append(i*j*k)
        ## basis function /phi
        phi=[]
        for i in w:
            phi_i=i/sum(w)
            phi.append(phi_i)
        #phi=np.array(phi)
        ## cost function C
        C=self.p.dot(np.array(phi))
        
        return C,phi
    
    def learn(self, state, reward, state_next, phi_list):
        # get s1,s2,s3(n,n+1), phi_list
        s1_not,s2_not,s3_not=state
        s1_up, s2_up, s3_up =state_next
        phi_list=np.array(phi_list)
        
        # calculate C(sn,pn)
        C_n,phi_n=self.cost_phi(state)
        
        # calculate C(sn+1,pn)
        C_n_1,phi=self.cost_phi(state_next)

        # learning rate(N=100)
        for i in range(len(self.alpha)):
            self.alpha[i]=(0.1/(1+100*sum(phi_list[:-1,i])))

        # eligibility trace
        for i in range(len(self.z)):
            self.z[i]=self.lam*np.exp(-self.beta*s3_not)*self.z[i]+phi_n[i]

        # update p
        for i in range(len(self.p)):
            self.p[i]=self.p[i]+self.alpha[i]*(reward+np.exp(-self.beta*s3_not)*C_n_1-C_n)*self.z[i]

        # return [phi^i] for phi_list
        return phi
            
    
    def c_up_c_not(self,tier,file,requestsDF):
        fileNo=int(file['No.'])
        requested=requestsDF.loc[requestsDF['request']==1].loc[requestsDF['No.'].isin(tier['No.'])]
        # if file already in tier
        if fileNo in list(tier['No.']):
            if len(tier)==0:
                s1_not=0
                s2_not=1000000  ## set it to be large enough
                s3_not=0
            else:
                ### s1_not of tier(not)
                s1_not=tier['temp'].mean()
                ### s2_not of tier(not)
                s2_not=sum([tier['temp'][i]*tier['weight'][i] for i in tier.index])/len(tier)
                ### s3_not of tier(not)
                response_times=list(requested['response'])
                s3_not=sum(response_times)/len(response_times)
                #s3_not=0
            ### C_not
            C_not,_=self.cost_phi([s1_not,s2_not,s3_not])

            tier_up=tier.drop(tier.loc[tier['No.']==fileNo].index).reset_index(drop=True)
            if len(tier_up)==0:
                s1_up=0
                s2_up=1000000  ## set it to be large enough
                s3_up=0
            else:
                ### s1_up of tier_up(up)
                s1_up=tier_up['temp'].mean()
                ### s2_up of tier_up(up)
                s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
                ### s3_up of tier_up(up)
                requested_up=requested.drop(requested.loc[requested['No.']==fileNo].index)
                response_times=list(requested_up['response'])
                if len(response_times)==0:
                    s3_up=0
                else:
                    s3_up=sum(response_times)/len(response_times)
                #s3_up=0
            ### C_up
            C_up,_=self.cost_phi([s1_up,s2_up,s3_up])
        # if file previously not in tier
        else:
            if len(tier)==0:
                s1_not=0
                s2_not=1000000  ## set it to be large enough
                s3_not=0
            else:
                ### s1_not of tier(not)
                s1_not=tier['temp'].mean()
                ### s2_not of tier(not)
                s2_not=sum([tier['temp'][i]*tier['weight'][i] for i in tier.index])/len(tier)
                ### s3_not of tier(not)
                response_times=list(requested['response'])
                if len(response_times)==0:
                    s3_not=0
                else:
                    s3_not=sum(response_times)/len(response_times)
                #s3_not=0
            ### C_not
            C_not,_=self.cost_phi([s1_not,s2_not,s3_not])

            tier_up=pd.concat([tier,file],ignore_index=True)
            if len(tier_up)==0:
                s1_up=0
                s2_up=1000000  ## set it to be large enough
                s3_up=0
            else:
                ### s1_up of tier_up(up)
                s1_up=tier_up['temp'].mean()
                ### s2_up of tier_up(up)
                s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
                ### s3_up of tier_up(up)
                request_file=requestsDF.loc[requestsDF['request']==1].loc[requestsDF['No.']==fileNo]
                requested_up=pd.concat([requested,request_file],ignore_index=True)
                response_times=list(requested_up['response'])
                s3_up=sum(response_times)/len(response_times)
                #s3_up=0
            ### C_up
            C_up,_=self.cost_phi([s1_up,s2_up,s3_up])

        return C_not,s1_not,C_up,s1_up
