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