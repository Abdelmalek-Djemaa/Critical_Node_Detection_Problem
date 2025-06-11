import torch
import time
from models import GNCR
from utils import (
    nx_to_pyg,
    load_trained_model,
    load_graph_dataset,
    load_criticality_scores,
)
from metrics import top_n_accuracy

# Load dataset

# ----------- Synthetic Graphs ----------- 
dataset_name = "500_synthetic.mtx"
# dataset_name = "1000_synthetic.mtx"
# dataset_name = "3000_synthetic.mtx"
# dataset_name = "5000_synthetic.mtx"
# dataset_name = "10000_synthetic.mtx"

# ----------- Real-world Graphs -----------

# dataset_name = "bio-yeast.mtx"
# dataset_name = "power-US-Grid.mtx"
# dataset_name = "wiki-Vote.mtx"
# dataset_name = "cit-DBLP.mtx"

# Load the graph
G = load_graph_dataset(f"data/{dataset_name}")

# Load criticality scores

dataset_name = dataset_name.replace(".mtx", "")
y_true = load_criticality_scores(f"{dataset_name}_criticality_scores.pkl")
y_true = torch.tensor(y_true, dtype=torch.float32)

# Convert graph to PyG format
data = nx_to_pyg(G)

# Define model paths
model_paths = [
    "models/model_SAGE_pl.pth",
    "models/model_SAGE_plc.pth",
]

hidden_dim = 32
results = {}

# Function to evaluate models and measure prediction time
def evaluate_models(model_paths, data, y_true):
    for model_path in model_paths:
        model = GNCR(hidden_dim)
        model = load_trained_model(model, model_path)
        model.eval()

        # Measure prediction time
        start_time = time.time()
        with torch.no_grad():
            y_pred = model(data)
        end_time = time.time()

        prediction_time = end_time - start_time

        # Calculate top-N accuracy (N=5)
        accuracy = top_n_accuracy(y_pred, y_true, N=5)

        # Store results
        results[model_path] = (accuracy, prediction_time)

# Run evaluation
evaluate_models(model_paths, data, y_true)

# Print results
print("\nTop-5% Accuracy Results:")
for model_path, (acc, pred_time) in results.items():
    print(f"{model_path}: {acc * 100:.2f}% | Prediction Time: {pred_time:.6f} seconds")
