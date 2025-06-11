import os
from utils import (
    generate_power_law,
    generate_power_law_cluster,
    save_graph_as_mtx,
    load_graph_dataset,
    compute_criticality_scores,
    save_criticality_scores
)
from metrics import compute_effective_resistance

os.makedirs("data", exist_ok=True)

# ----------- Synthetic Graphs -----------

sizes = [500, 1000, 3000, 5000,10000]
m = 4
p = 0.4 

for size in sizes:
    G_pl = generate_power_law(n=size, m=m)
    pl_name = f"{size}_synthetic"
    save_graph_as_mtx(G_pl, f"data/{pl_name}.mtx")
    scores_pl = compute_criticality_scores(G_pl, compute_effective_resistance)
    save_criticality_scores(scores_pl, f"{pl_name}_criticality_scores.pkl")

    G_plc = generate_power_law_cluster(n=size, m=m, p=p)
    plc_name = f"{size}_synthetic"
    save_graph_as_mtx(G_plc, f"data/{plc_name}.mtx")
    scores_plc = compute_criticality_scores(G_plc, compute_effective_resistance)
    save_criticality_scores(scores_plc, f"{plc_name}_criticality_scores.pkl")

# ----------- Real-world Graphs -----------

real_world_graphs = [
    "bio-yeast.mtx",
    "power-US-Grid.mtx",
    "wiki-Vote.mtx",
    "cit-DBLP.mtx"
]

for dataset_name in real_world_graphs:
    graph_path = f"data/{dataset_name}"
    G_real = load_graph_dataset(graph_path)
    base_name = dataset_name.replace(".mtx", "") 
    scores_real = compute_criticality_scores(G_real, compute_effective_resistance)
    save_criticality_scores(scores_real, f"{base_name}_criticality_scores.pkl")