import os, sys, getopt, pdb
import time
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from termcolor import colored, cprint

pd.set_option("max_rows", 100)

import warnings

from IPython.display import clear_output # Used to clear the ouput of a Jupyter cell.

from progressbar import *

## plot of number of transfers, temp46-fixgenreq
transfer_list_RL_zero=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_zeroinit123_temp46_tempdesc10.npy')
transfer_list_RL_10100=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_10100init123_temp46_tempdesc10.npy')
transfer_list_RL_all=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_allinit123_temp46_tempdesc10.npy')
transfer_list_rule1=np.load('./transfer_list_3tier_1000/transfer_list_rule1_1000_123_fixgenreq_temp46_tempdesc10.npy')
transfer_list_rule2=np.load('./transfer_list_3tier_1000/transfer_list_rule2_1000_123_fixgenreq_temp46_tempdesc10.npy')
transfer_list_rule3=np.load('./transfer_list_3tier_1000/transfer_list_rule3_1000_123_fixgenreq_temp46_tempdesc10.npy')

plt.figure(1,figsize=(12,10), dpi=320)

plt.subplot(2,2,1)
plt.plot(np.array(transfer_list_RL_zero)[:,0],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,0],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,0],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,0],color='green')
plt.plot(np.array(transfer_list_rule2)[:,0],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,0],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier1 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,2)
plt.plot(np.array(transfer_list_RL_zero)[:,2],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,2],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,2],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,2],color='green')
plt.plot(np.array(transfer_list_rule2)[:,2],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,2],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier2 to tier3')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,3)
plt.plot(np.array(transfer_list_RL_zero)[:,4],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,4],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,4],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,4],color='green')
plt.plot(np.array(transfer_list_rule2)[:,4],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,4],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier3 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,4)
plt.plot(np.array(transfer_list_RL_zero)[:,6],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,6],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,6],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,6],color='green')
plt.plot(np.array(transfer_list_rule2)[:,6],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,6],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier2 to tier1')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.savefig('./Num_transfer_RL&rule123_1000_temp46_fixgenreq.png',format='png',dpi=320)
plt.show()



## plot of number of transfers, temp01-fixgenreq
transfer_list_RL_zero=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_zeroinit123_temp01_tempdesc10.npy')
transfer_list_RL_10100=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_10100init123_temp01_tempdesc10.npy')
transfer_list_RL_all=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixgenreq_allinit123_temp01_tempdesc10.npy')
transfer_list_rule1=np.load('./transfer_list_3tier_1000/transfer_list_rule1_1000_123_fixgenreq_temp01_tempdesc10.npy')
transfer_list_rule2=np.load('./transfer_list_3tier_1000/transfer_list_rule2_1000_123_fixgenreq_temp01_tempdesc10.npy')
transfer_list_rule3=np.load('./transfer_list_3tier_1000/transfer_list_rule3_1000_123_fixgenreq_temp01_tempdesc10.npy')

plt.figure(1,figsize=(12,10), dpi=320)

plt.subplot(2,2,1)
plt.plot(np.array(transfer_list_RL_zero)[:,0],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,0],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,0],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,0],color='green')
plt.plot(np.array(transfer_list_rule2)[:,0],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,0],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier1 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,2)
plt.plot(np.array(transfer_list_RL_zero)[:,2],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,2],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,2],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,2],color='green')
plt.plot(np.array(transfer_list_rule2)[:,2],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,2],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier2 to tier3')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,3)
plt.plot(np.array(transfer_list_RL_zero)[:,4],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,4],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,4],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,4],color='green')
plt.plot(np.array(transfer_list_rule2)[:,4],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,4],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier3 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,4)
plt.plot(np.array(transfer_list_RL_zero)[:,6],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,6],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,6],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,6],color='green')
plt.plot(np.array(transfer_list_rule2)[:,6],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,6],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier2 to tier1')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.savefig('./Num_transfer_RL&rule123_1000_temp01_fixgenreq.png',format='png',dpi=320)
plt.show()



## plot of number of transfers, temp46-randunireq
transfer_list_RL_zero=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixnumrandpoireq_zeroinit123_temp46_tempdesc5_0.01.npy')
transfer_list_RL_10100=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixnumrandpoireq_10100init123_temp46_tempdesc5_0.01.npy')
transfer_list_RL_all=np.load('./transfer_list_3tier_1000/transfer_list_RL_1000_fixnumrandpoireq_allinit123_temp46_tempdesc5_0.01.npy')
transfer_list_rule1=np.load('./transfer_list_3tier_1000/transfer_list_rule1_1000_123_fixnumrandpoireq_temp46_tempdesc5_0.01.npy')
transfer_list_rule2=np.load('./transfer_list_3tier_1000/transfer_list_rule2_1000_123_fixnumrandpoireq_temp46_tempdesc5_0.01.npy')
transfer_list_rule3=np.load('./transfer_list_3tier_1000/transfer_list_rule3_1000_123_fixnumrandpoireq_temp46_tempdesc5_0.01.npy')

plt.figure(1,figsize=(12,10), dpi=320)

plt.subplot(2,2,1)
plt.plot(np.array(transfer_list_RL_zero)[:,0],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,0],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,0],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,0],color='green')
plt.plot(np.array(transfer_list_rule2)[:,0],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,0],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier1 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,2)
plt.plot(np.array(transfer_list_RL_zero)[:,2],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,2],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,2],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,2],color='green')
plt.plot(np.array(transfer_list_rule2)[:,2],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,2],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being upgraded from tier2 to tier3')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,3)
plt.plot(np.array(transfer_list_RL_zero)[:,4],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,4],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,4],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,4],color='green')
plt.plot(np.array(transfer_list_rule2)[:,4],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,4],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier3 to tier2')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.subplot(2,2,4)
plt.plot(np.array(transfer_list_RL_zero)[:,6],color='purple')
plt.plot(np.array(transfer_list_RL_10100)[:,6],color='blue')
plt.plot(np.array(transfer_list_RL_all)[:,6],color='deepskyblue')
plt.plot(np.array(transfer_list_rule1)[:,6],color='green')
plt.plot(np.array(transfer_list_rule2)[:,6],color='orange')
plt.plot(np.array(transfer_list_rule3)[:,6],color='red')
plt.xlabel('timesteps')
plt.ylabel('Number of transfer files')
plt.ylim(0,150)
plt.title('Number of files being downgraded from tier2 to tier1')
plt.legend(['RL_st','RL_dt','RL_ft','Rule-based 1','Rule_based 2','Rule_based 3'],loc='upper right')

plt.savefig('./Num_transfer_RL&rule123_1000_temp46_randunireq.png',format='png',dpi=320)
plt.show()