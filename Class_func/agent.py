#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, getopt, pdb

import numpy as np
import pandas as pd
import random

##TD() agent 
class TDAgent():
    
    def __init__(self,n_states,p_init,beta,lam,a1,a2,a3,b1,b2,b3):
        self.n_states = n_states
        # parameters of cost-to-go func
        self.p = np.array(p_init)
        # parameters of FRB func
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
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
            print(self.p[i])

        # return [phi^i] for phi_list
        return phi
            
    
    def c_up_c_not(self,tier,file):
        fileNo=int(file['No.'])
        if fileNo in list(tier['No.']):
            ### s1_not of tier(not)
            s1_not=tier['temp'].mean()
            ### s2_not of tier(not)
            s2_not=sum([tier['temp'][i]*tier['weight'][i] for i in tier.index])/len(tier)
            ### s3_not of tier(not)
            s3_not=0
            ### C_not
            C_not,_=self.cost_phi([s1_not,s2_not,s3_not])

            tier_up=tier.drop(tier.loc[tier['No.']==fileNo].index).reset_index(drop=True)
            ### s1_up of tier_up(up)
            s1_up=tier_up['temp'].mean()
            ### s2_up of tier_up(up)
            s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
            ### s3_up of tier_up(up)
            s3_up=0
            ### C_up
            C_up,_=self.cost_phi([s1_up,s2_up,s3_up])
        else:
            ### s1_not of tier(not)
            s1_not=tier['temp'].mean()
            ### s2_not of tier(not)
            s2_not=sum([tier['temp'][i]*tier['weight'][i] for i in tier.index])/len(tier)
            ### s3_not of tier(not)
            s3_not=0
            ### C_not
            C_not,_=self.cost_phi([s1_not,s2_not,s3_not])

            tier_up=pd.concat([tier,file],ignore_index=True)
            ### s1_up of tier_up(up)
            s1_up=tier_up['temp'].mean()
            ### s2_up of tier_up(up)
            s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
            ### s3_up of tier_up(up)
            s3_up=0
            ### C_up
            C_up,_=self.cost_phi([s1_up,s2_up,s3_up])

        return C_not,s1_not,C_up,s1_up
