import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np


def compute_effective_resistance(graph):
    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]
    N = graph.number_of_nodes()
    return (2 / (N - 1)) * np.sum(1 / eigenvalues)

def pairwise_ranking_loss(y_pred, y_true):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    # Compute pairwise score and label differences
    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)

    # Keep only upper triangle (i < j) to avoid redundancy and self-pairs
    mask = torch.triu(torch.ones_like(diff_true), diagonal=1).bool()
    diff_pred = diff_pred[mask]
    diff_true = diff_true[mask]

    # Target: 1 if y_i > y_j, 0 otherwise
    target = (diff_true > 0).float()

    # Compute sigmoid cross-entropy (i.e., binary cross-entropy)
    loss = F.binary_cross_entropy_with_logits(diff_pred, target)

    return loss

def top_n_accuracy(y_pred, y_true, N=5):
    top_n = int(len(y_true) * (N / 100))
    top_pred = torch.argsort(y_pred.squeeze(), descending=True)[:top_n]
    top_true = torch.argsort(y_true.squeeze(), descending=True)[:top_n]
    return len(set(top_pred.tolist()) & set(top_true.tolist())) / top_n
