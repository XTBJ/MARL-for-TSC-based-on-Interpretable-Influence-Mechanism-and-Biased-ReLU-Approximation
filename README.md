# Multi-agent Reinforcement Learning for Traffic Signal Control based on Interpretable Influence Mechanism and Biased ReLU Approximation

Traffic signal control is important in intelligent transportation systems, of which cooperative control is difficult to realize yet vital. Many methods model multi-intersection traffic networks as grids and address the problem using multi-agent reinforcement learning (RL). Despite these existing studies, there is an opportunity to further enhance our understanding of the connectivity and globality of the traffic networks by capturing the spatiotemporal traffic information with efficient neural networks in deep RL. In this work, we propose a novel multi-agent actor-critic framework based on an interpretable influence mechanism with a centralized learning and decentralized execution method.

Specifically, we first construct an actor-critic framework, for which the piecewise linear neural network(PWLNN), named biased ReLU (BReLU), is used as the function approximator to obtain a more accurate and theoretically grounded approximation. Then, to model the relationships among agents in multi-intersection scenarios, we introduce an interpretable influence mechanism based on efficient hinging hyperplanes neural network (EHHNN), which derives weights by analysis of variance (ANOVA) decomposition among agents and extracts spatiotemporal dependencies of the traffic features. Finally, our proposed framework is validated on two synthetic traffic networks and a real road network to coordinate signal control between intersections, achieving lower traffic delays across the entire traffic network compared with state-of-the-art (SOTA) performance.

## About the Project

This work aims to develop an end-to-end solution for the multi-intersection traffic signal control problem, which can send phase commands to the traffic signal controllers to minimize the total vehicle waiting time of the global traffic network, and is divided in the following components:

1. SUMO Environment setup.
2. An interpretable EHHNN.
3. Multi-agent BReLU actor-critic framework for optimization.

## Requirements

To install requirements:

`pip install -r `

## How to Run

### Training a New Agent

In order to train a new agent use the following command:

`python main.py --method_name IPPO`

This will start training an agent with the default parameters, and the checkpoints will be written to `checkpoints/<env_name>/` and the metrics will bee recorded into `results/<env_name>/`.

For the env_name, we provide three different traffic network: 
1. grid 
2. noneuclidean
3. cologne

Besides, for the method_name, we provide 5 kinds of multi-agent reinforcement learning methods, including:
1. Independent Deep Q Network (IDQN)
2. Independent Proximal Policy Optimization (IPPO)
3. Multi-Agent Deep deterministic Policy Gradient (MADDPG)
4. Graph Convolution Reinforcement Learning (DGN)
5. **Multi-agent BReLU Actor-Critic -- our proposed method (BRGEHH)**

### Results

We evaluate our proposed method on two synthetic traffic networks and a real road network in Cologne, as shown in the following figures.

<div style="text-align:center">
  <img src="results/sythetic.png" width="500" height="250">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 1. Sythetic Traffic networks </center> 

<div style="text-align:center">
  <img src="results/cologne.png" width="500" height="300">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 2. The real road network </center> 

Table 1 represents a comparison of the performance of BReLU-EHH with seven other algorithms, including three traditional algorithms and four RL methods.


<center style="font-size:14px;color:#C0C0C0">Table 1. Experimental performance of different traffic networks </center> 

![](/results/table.png)

- **Experimental results on the 5 $\times$ 5 traffic grid**

<div style="text-align:center">
  <img src="results/train_grid.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 3. Performance Comparison on the 5×5 Synthetic Traffic Grid throughout the entire training process. </center> 


<div style="text-align:center">
  <img src="results/test_grid.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 4. Performance of all methods on the 5×5 Synthetic Traffic Grid during the test process. </center> 

- **Experimental results on the Non-Euclidean traffic network**

<div style="text-align:center">
  <img src="results/train_non.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 5. Performance Comparison on the Non-Euclidean Synthetic Traffic Network throughout the entire training process. </center> 

<div style="text-align:center">
  <img src="results/test_non.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 6. Performance Comparison on the Non-Euclidean Synthetic Traffic Network during the test process. </center> 

- **Experimental results on the real road network**

<div style="text-align:center">
  <img src="results/train_cologne.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 7. Performance Comparison on the Real Road Network throughout the entire training process. </center> 

<div style="text-align:center">
  <img src="results/test_cologne.png">
</div>
<center style="font-size:14px;color:#C0C0C0">Figure 8. Performance Comparison on the Real Road Network during the test process. </center> 