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

def pairwise_ranking_loss(y_pred, y_true, max_pairs=2000):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    n = len(y_true)
    
    # Sample random pairs to prevent O(N^2) memory explosion
    idx_i = torch.randint(0, n, (max_pairs,))
    idx_j = torch.randint(0, n, (max_pairs,))
    
    diff_pred = y_pred[idx_i] - y_pred[idx_j]
    diff_true = y_true[idx_i] - y_true[idx_j]
    
    target = (diff_true > 0).float()
    return F.binary_cross_entropy_with_logits(diff_pred, target)

def top_n_accuracy(y_pred, y_true, N=5):
    y_pred, y_true = y_pred.view(-1), y_true.view(-1)
    top_k = max(1, int(len(y_true) * (N / 100)))
    
    _, top_pred_idx = torch.topk(y_pred, top_k)
    _, top_true_idx = torch.topk(y_true, top_k)
    
    itst = set(top_pred_idx.tolist()) & set(top_true_idx.tolist())
    return len(itst) / top_k