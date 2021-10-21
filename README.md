# HSM-RL: Hierarchical Storage Management using Reinforcement Learning

## Introduction

Online data migration policy for Hierarchical Storage System based on Reinforcement Learning. 

In a Hierarchical (Multi-tier) Storage System (HSS), file transfers are controlled by the Data Migration Policy, thus a well-defined policy is crucial for keeping the HSS ordered and well-performed. We use reinforcement learning knowledages to train a continuously-updating (online) migration policy.

<div  align="center">
<img src="https://github.com/JSFRi/HSM-RL/blob/main/HSS.png" width = "300" height = "200" alt="HSS" />
</div>

The entire theory and algorithm are explained in the article.

## Codes

Core functions of RL are in the folder Class_func/

'env.py' defines the environments class that used to define the environmental variables of a tier

'agent.py' defines the RL agent class that creates RL agent for a tier

'req_generator.py' defines requests generators that used to generate simulated requested

### Simulation setups

Folder simu_experiments/ contains implementations of simulations of a system containes three tiers(fast,medium,slow) and 1000 files using RL-based policies and rule-based policies. Corresponding with the section 6.1 in the article.

### Cloud-based setups
Folder ansible_client/ includes codes to start different tier instances using ansible. You can start two or three tiers instances by using 'start_instances.py' or 'start_instances_3tier.py'. They should be modified according to your local environments.

Celery workers are included in folder celery_workers/, designed for fast/medium/slow tiers. One should modify them according to your local environments.

Folder cloud_experiments/ contains implementations experiments on cloud-distributed system using RL-based policies and rule-based policies. Corresponding with the section 6.2 in the article.

## Supplements
Video Animations about how file distirbutions change over timesteps: see MP4 files named 'Heatmaps_xxx.mp4'

Article: 

