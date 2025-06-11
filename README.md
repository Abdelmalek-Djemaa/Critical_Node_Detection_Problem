# Critical node detection problem 

This repository provides a framework for evaluating the robustness of graphs and identifying critical nodes using various graph robustness metrics and ranking loss-based methods. The goal is to assess the resilience of networks to node removals and rank nodes based on their criticality in maintaining the overall network structure.

## Table of Contents

1. [Introduction](#introduction)
2. [Graph Robustness Metric](#graph-robustness-metric)
    - [Effective Graph Resistance (EGR)](#effective-graph-resistance-egr)
3. [Conventional Approach for Identifying Critical Nodes](#conventional-approach-for-identifying-critical-nodes)
4. [Ranking Loss](#ranking-loss)
5. [Evaluation Metrics: Top-N% Accuracy](#evaluation-metrics-top-n-accuracy)

## Introduction

This repository contains implementations for graph robustness metrics and a criticality ranking approach based on the robustness of the graph after node removal. The framework helps identify important nodes in a graph and evaluate the model's performance using Top-N% accuracy.

## Graph Robustness Metrics

### Effective Graph Resistance (EGR)

The Effective Graph Resistance (EGR) is a metric that measures the vulnerability of a graph based on the eigenvalues of its Laplacian matrix. The formula for EGR is:

$$
R_g = \frac{2}{N-1} \sum_{i=1}^{N-c} \frac{1}{\lambda_i}
$$

Where:
- $N$ is the total number of nodes in the graph.
- $\lambda_i$ are the eigenvalues of the Laplacian matrix of the graph.
- $c$ is the number of eigenvalues considered in the calculation.


## Conventional Approach for Identifying Critical Nodes

### Input:
- Graph $G$ with $V$ nodes.

### Output:
- Node critical scores.

### Steps:
1. For each node $n$ in $V$:
   - Remove node $n$ from the graph $G$.
   - Compute the robustness metric of the residual graph $(G - n)$.
   - Assign a criticality score to node $n$.
   
2. Rank the nodes based on the computed criticality scores.
3. The top ranks correspond to the most critical nodes.
4. Return the top $N\%$ of the most critical nodes.

## Ranking Loss

The ranking loss evaluates the quality of predicted node rankings by comparing pairs of nodes using binary cross-entropy.

Let:

- $s_{ij} = \hat{y}^i - \hat{y}^j$  the difference in predicted scores
- $t_{ij} = 1$ if $y^i > y^j$, otherwise $t_{ij} = 0$

Then, the pairwise loss is:

$$
\ell_{ij} = -t_{ij} \cdot \log(\sigma(s_{ij})) - (1 - t_{ij}) \cdot \log(1 - \sigma(s_{ij}))
$$

Where $\sigma(x) = \frac{1}{1 + \exp(-x)}$ is the sigmoid function.

The final loss is the average over all such pairs:

$$
L = \frac{1}{|P|} \sum_{i < j} \ell_{ij}
$$

$\sigma(\hat{y}^i - \hat{y}^j)$ and the label $t_{ij} \in \{0, 1\}$.

This loss encourages the model to assign higher scores to more critical nodes in accordance with the true ranking.


## Evaluation Metrics: Top-N% Accuracy

The **Top-N% Accuracy** metric is used to measure the accuracy of the predicted critical nodes compared to a ground-truth baseline. It is defined as:

$$
\text{Top-N\% Accuracy} = \frac{\left| \{\text{Predicted Top-N\% nodes}\} \cap \{\text{True Top-N\% nodes}\} \right|}{|V| \times (N/100)}
$$

Where:
- $|V|$ is the total number of nodes in the graph.
- $N$ is the percentage band (e.g., Top-5%).
- $\cap$ denotes the intersection between the predicted and true Top-N% sets.
