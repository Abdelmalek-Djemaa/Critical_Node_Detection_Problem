import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.io import mmread
from scipy.io import mmwrite
from tqdm import tqdm
from io import StringIO
import pickle
import os


DATA_DIR = 'data/'

def compute_criticality_scores(graph, metric):
    scores = []
    original_metric_value = metric(graph)  # Compute once
    for node in tqdm(graph.nodes(), desc="Computing Criticality Scores"):
        subgraph = graph.copy()
        subgraph.remove_node(node)
        score = metric(subgraph) - original_metric_value 
        scores.append(score)
    return scores

def save_criticality_scores(scores, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(scores, f)
    print(f"Criticality scores saved to {filepath}")

def load_criticality_scores(filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        scores = pickle.load(f)
    print(f"Criticality scores loaded from {filepath}")
    return scores


def generate_power_law(n, m):
    return nx.barabasi_albert_graph(n, m)

def generate_power_law_cluster(n, m, p):
    return nx.powerlaw_cluster_graph(n, m, p) 

def generate_node_features(nx_graph):
    # Get node degrees
    degrees = torch.tensor([deg for _, deg in nx_graph.degree()], dtype=torch.float32)

    # Normalize degrees
    max_degree = degrees.max()
    normalized_degrees = degrees / max_degree if max_degree > 0 else degrees

    # Return as a (num_nodes, 1) feature matrix
    return normalized_degrees.view(-1, 1)


def nx_to_pyg(nx_graph, features=None):
    pyg_data = from_networkx(nx_graph)
    
    if features is not None:

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.clone().detach()
        
        pyg_data.x = features.view(-1, 1)
    else:
        # Use normalized degree + 1 as node features
        pyg_data.x = generate_node_features(nx_graph)
    
    return pyg_data




def load_trained_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"No pretrained model found at {model_path}. Starting fresh.")
    return model


from scipy.io import mmread
from io import StringIO
import networkx as nx

def load_graph_dataset(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter out comment lines
    comment_lines = [line for line in lines if line.startswith('%')]
    data_lines = [line for line in lines if not line.startswith('%') and line.strip()]

    # Parse all non-comment lines into triples
    entries = []
    max_index = 0
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue  # skip invalid lines
        i, j = int(parts[0]), int(parts[1])
        val = int(parts[2]) if len(parts) > 2 else 1
        entries.append((i, j, val))
        max_index = max(max_index, i, j)

    # Rebuild the shape line
    n = max_index
    nnz = len(entries)
    shape_line = f"{n} {n} {nnz}\n"

    # Build new .mtx content
    new_lines = comment_lines + [shape_line] + [f"{i} {j} {val}\n" for (i, j, val) in entries]
    matrix = mmread(StringIO(''.join(new_lines)))

    try:
        G = nx.from_scipy_sparse_array(matrix)
    except AttributeError:
        G = nx.from_scipy_sparse_matrix(matrix)
    return G




def save_graph_as_mtx(G, filename):
    A = nx.adjacency_matrix(G)
    mmwrite(filename, A)
