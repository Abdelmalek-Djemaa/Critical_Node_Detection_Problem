import os
import random
import pickle
from utils import compute_criticality_scores, generate_power_law, generate_power_law_cluster
from metrics import compute_effective_resistance

# === SETTINGS ===
DATA_DIR = "train_data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_graph_and_scores(G, scores, filename):
    with open(os.path.join(DATA_DIR, filename), "wb") as f:
        pickle.dump({"graph": G, "scores": scores}, f)
    print(f"Saved: {filename}")

def generate_and_save_graphs(graph_generator, metric, prefix, num_graphs=30, min_size=100, max_size=1000):
    for i in range(num_graphs):
        size = random.randint(min_size, max_size)

        # Randomize parameters
        m = random.randint(1, min(size - 1, 10))  # number of edges to attach
        p = round(random.uniform(0.1, 0.9), 2)    # clustering probability

        if prefix == "pl":
            G = generate_power_law(size, m)
            filename = f"{prefix}_{i}.pkl"
        elif prefix == "plc":
            G = generate_power_law_cluster(size, m, p)
            filename = f"{prefix}_{i}.pkl"
        else:
            raise ValueError("Unknown prefix.")

        scores = compute_criticality_scores(G, metric)
        save_graph_and_scores(G, scores, filename)

def main():
    print("Generating Power Law graphs...")
    generate_and_save_graphs(lambda size: None, compute_effective_resistance, "pl")

    print("Generating Power Law Cluster graphs...")
    generate_and_save_graphs(lambda size: None, compute_effective_resistance, "plc")

if __name__ == "__main__":
    main()
