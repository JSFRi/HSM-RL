# HSM-RL: Hierarchical Storage Management using Reinforcement Learning

## Introduction

Online data migration policy for Hierarchical Storage System based on Reinforcement Learning. 

In a Hierarchical (Multi-tier) Storage System (HSS), file transfers are controlled by the Data Migration Policy, thus a well-defined policy is crucial for keeping the HSS ordered and well-performed. We use reinforcement learning knowledages to train a continuously-updating (online) migration policy.

<div  align="center">
<img src="https://github.com/JSFRi/HSM-RL/blob/main/HSS.png" width = "300" height = "200" alt="HSS" />
</div>

## Methods

For each storage tier, we build a RL agent to learn the cost function of this tier, and the migration policy is defined by criterion: For a file in tier i being requested, it will transferred to tier j if:

$$\frac{1}{2}$$

$$ {C_{up}^i\cdot\tilde{s}_i^i+C_{up}^j\cdot\tilde{s}_j^i < C_{not}^i\cdot s_i^i+C_{not}^j\cdot{s}_j^i} $$


'Workflow Chart:'
![image](https://github.com/JSFRi/HSM-RL/blob/main/Flow_Chart.png)

## Codes

<div  align="center">
<img src="https://github.com/JSFRi/HSM-RL/blob/main/heatmap_160.png" width = "300" height = "200" alt="HSS" />
</div>

## Supplements
Slides: https://drive.google.com/file/d/1ZUtKDQc8nF0HM2CJXTVQEzcWTZa2D7za/view?usp=sharing

Reference:

Vengerov, D. A reinforcement learning framework for online data migration in hierarchical storage systems. J Supercomput 43, 1–19 (2008). https://doi.org/10.1007/s11227-007-0135-3
