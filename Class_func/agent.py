#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, getopt, pdb

import numpy as np
import pandas as pd
import random

##TD() agent 
class TDAgent():
    
    def __init__(self,n_states,p1_init,p2_init,beta,lam,a1,a2,a3,b1,b2,b3):
        self.n_states = n_states
        # parameters of cost-to-go func
        self.p1 = p1_init
        self.p2 = p2_init
        # parameters of FRB func
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        # parameters of TD
        self.beta=beta
        self.lam = lam
        # eligibility trace
        self.z1 = 0
        self.z2 = 0
        
    def act(self, state):
        # Random
        return np.random.choice(self.n_actions)
    
    def learn(self, state, reward, state_next, phi1s_list, phi2s_list):
        # get s1,s2,s3(n,n+1)
        s1_not,s2_not,s3_not=state
        s1_up, s2_up, s3_up =state_next
        
        # calculate C(sn,pn)
        w1s_n=(1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1/(1+self.a3*np.exp(-self.b3*s3_not)))
        w2s_n=(1-1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_not)))
        phi1s_n=w1s_n/(w1s_n+w2s_n)
        phi2s_n=w2s_n/(w1s_n+w2s_n)
        C_n=self.p1*phi1s_n+self.p2*phi2s_n
        
        # calculate C(sn+1,pn)
        w1s=(1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1/(1+self.a3*np.exp(-self.b3*s3_up)))
        w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_up)))
        phi1s=w1s/(w1s+w2s)
        phi2s=w2s/(w1s+w2s)
        C_n_1=self.p1*phi1s+self.p2*phi2s

        # learning rate(N=50)
        alpha1=0.1/(1+50*sum(phi1s_list[:-1]))
        alpha2=0.1/(1+50*sum(phi2s_list[:-1]))

        # eligibility trace
        self.z1=self.lam*np.exp(-self.beta*s3_not)*self.z1+phi1s_n
        self.z2=self.lam*np.exp(-self.beta*s3_not)*self.z2+phi2s_n

        # update p1,p2
        self.p1=self.p1+alpha1*(reward+np.exp(-self.beta*s3_not)*C_n_1-C_n)*self.z1
        self.p2=self.p2+alpha2*(reward+np.exp(-self.beta*s3_not)*C_n_1-C_n)*self.z2

        # return phi1s,phi2s for phi_list
        return phi1s,phi2s
        
    def count_phi(self,state):
        s1,s2,s3=state
        w1s=(1/(1+self.a1*np.exp(-self.b1*s1)))*(1/(1+self.a2*np.exp(-self.b2*s2)))*(1/(1+self.a3*np.exp(-self.b3*s3)))
        w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1)))*(1-1/(1+self.a2*np.exp(-self.b2*s2)))*(1-1/(1+self.a3*np.exp(-self.b3*s3)))
        phi1s=w1s/(w1s+w2s)
        phi2s=w2s/(w1s+w2s)
        return phi1s,phi2s
    
    
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
            w1s=(1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1/(1+self.a3*np.exp(-self.b3*s3_not)))
            w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_not)))
            phi1s=w1s/(w1s+w2s)
            phi2s=w2s/(w1s+w2s)
            C_not=self.p1*phi1s+self.p2*phi2s

            tier_up=tier.drop(tier.loc[tier['No.']==fileNo].index).reset_index(drop=True)
            ### s1_up of tier_up(up)
            s1_up=tier_up['temp'].mean()
            ### s2_up of tier_up(up)
            s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
            ### s3_up of tier_up(up)
            s3_up=0
            ### C_up
            w1s=(1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1/(1+self.a3*np.exp(-self.b3*s3_up)))
            w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_up)))
            phi1s=w1s/(w1s+w2s)
            phi2s=w2s/(w1s+w2s)
            C_up=self.p1*phi1s+self.p2*phi2s
        else:
            ### s1_not of tier(not)
            s1_not=tier['temp'].mean()
            ### s2_not of tier(not)
            s2_not=sum([tier['temp'][i]*tier['weight'][i] for i in tier.index])/len(tier)
            ### s3_not of tier(not)
            s3_not=0
            ### C_not
            w1s=(1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1/(1+self.a3*np.exp(-self.b3*s3_not)))
            w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1_not)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_not)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_not)))
            phi1s=w1s/(w1s+w2s)
            phi2s=w2s/(w1s+w2s)
            C_not=self.p1*phi1s+self.p2*phi2s

            tier_up=pd.concat([tier,file],ignore_index=True)
            ### s1_up of tier_up(up)
            s1_up=tier_up['temp'].mean()
            ### s2_up of tier_up(up)
            s2_up=sum([tier_up['temp'][i]*tier_up['weight'][i] for i in tier_up.index])/len(tier_up)
            ### s3_up of tier_up(up)
            s3_up=0
            ### C_up
            w1s=(1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1/(1+self.a3*np.exp(-self.b3*s3_up)))
            w2s=(1-1/(1+self.a1*np.exp(-self.b1*s1_up)))*(1-1/(1+self.a2*np.exp(-self.b2*s2_up)))*(1-1/(1+self.a3*np.exp(-self.b3*s3_up)))
            phi1s=w1s/(w1s+w2s)
            phi2s=w2s/(w1s+w2s)
            C_up=self.p1*phi1s+self.p2*phi2s

        return C_not,s1_not,C_up,s1_up

