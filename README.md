# HSM-RL: Hierarchical Storage Management using Reinforcement Learning

## Introduction

Online data migration policy for Hierarchical Storage System based on Reinforcement Learning. 

In a Hierarchical (Multi-tier) Storage System (HSS), file transfers are controlled by the Data Migration Policy, thus a well-defined policy is crucial for keeping the HSS ordered and well-performed. We use reinforcement learning knowledages to train a continuously-updating (online) migration policy.

<div  align="center">
<img src="https://github.com/JSFRi/HSM-RL/blob/main/HSS.png" width = "300" height = "200" alt="HSS" />
</div>

## Methods

For each storage tier, we build a RL agent to learn the cost function of this tier, and the migration policy is defined by criterion: *For a file in tier i being requested, it will transferred to tier j if* 

<div  align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=C_{up}^i\cdot\tilde{s}_i^i&plus;C_{up}^j\cdot\tilde{s}_j^i&space;<&space;C_{not}^i\cdot&space;s_i^i&plus;C_{not}^j\cdot{s}_j^i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C_{up}^i\cdot\tilde{s}_i^i&plus;C_{up}^j\cdot\tilde{s}_j^i&space;<&space;C_{not}^i\cdot&space;s_i^i&plus;C_{not}^j\cdot{s}_j^i" title="C_{up}^i\cdot\tilde{s}_i^i+C_{up}^j\cdot\tilde{s}_j^i < C_{not}^i\cdot s_i^i+C_{not}^j\cdot{s}_j^i" /></a>
</div>

Then the cost functions are updated continually by TD-lambda algortihm

For more details about the methods please look at the referance article.


## Codes

Fondamental functions are in the folder Class_func/

'env.py' defines the environments class that used to define the environmental variables of a tier

'agent.py' defines the RL agent class that creates RL agent for a tier

'req_generator.py' defines requests generators that used to generate simulated requested

'RL(/Rule1/Rule2/Rule3)-randinit.py' imply simulations of a system containes three tiers(fast,medium,slow) and 1000 files using policies according to RL agent(/Naive rule1/2/3), for more details please see the slides in supplements.

Folder fast_slow_tier_ansible_client/ includes codes to start fast&slow tier instances using ansible.

## Supplements
Slides: https://drive.google.com/file/d/1ZUtKDQc8nF0HM2CJXTVQEzcWTZa2D7za/view?usp=sharing
Article: arxiv.....

