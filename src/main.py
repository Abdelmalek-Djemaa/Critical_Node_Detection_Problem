import torch
import os
from models import GNCR
from utils import load_graph_dataset, nx_to_pyg, load_criticality_scores
from metrics import top_n_accuracy
from pipeline import run_pipeline

# Define all model configurations to test
MODEL_CONFIGS = [
    {"name": "Base", "kwargs": {}},
    {"name": "JK", "kwargs": {"use_jk": True}},
    {"name": "Attention", "kwargs": {"use_attention": True}},
    {"name": "Attention+JK", "kwargs": {"use_attention": True, "use_jk": True}},
    {"name": "AttentionOnly", "kwargs": {"only_attention": True}},
]

os.makedirs("models", exist_ok=True)

# Get all available datasets
data_dir = "data"
datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".mtx")])

print("=" * 100)
print("Training Models and Testing on All Datasets")
print("=" * 100)

# Dictionary to store results: {model_name: {dataset: accuracy}}
all_results = {config["name"]: {} for config in MODEL_CONFIGS}

# ============================================================================
# PHASE 1: TRAIN ALL MODELS
# ============================================================================
print("\n[PHASE 1] TRAINING ALL MODELS")
print("-" * 100)

for config in MODEL_CONFIGS:
    model_name = config["name"]
    model_kwargs = config["kwargs"]
    model_path = f"models/gncr_{model_name.lower()}.pth"
    
    if not os.path.exists(model_path):
        print(f"\nTraining [{model_name}] with config: {model_kwargs}")
        run_pipeline("pl", model_path, epochs=50, model_kwargs=model_kwargs)
    else:
        print(f"[{model_name}] Model already trained (skipping)")

# ============================================================================
# PHASE 2: TEST ALL MODELS ON ALL DATASETS
# ============================================================================
print("\n" + "=" * 100)
print("[PHASE 2] TESTING ALL MODELS ON ALL DATASETS")
print("=" * 100)

for config in MODEL_CONFIGS:
    model_name = config["name"]
    model_kwargs = config["kwargs"]
    model_path = f"models/gncr_{model_name.lower()}.pth"
    
    print(f"\n[{model_name}] Testing on {len(datasets)} datasets...")
    print("-" * 100)
    
    model = GNCR(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    for dataset_name in datasets:
        try:
            score_file = dataset_name.replace(".mtx", "_criticality_scores.pkl")
            G = load_graph_dataset(f"{data_dir}/{dataset_name}")
            y_true = torch.tensor(load_criticality_scores(f"{score_file}"))
            data = nx_to_pyg(G)

            with torch.no_grad():
                y_pred = model(data)
                acc = top_n_accuracy(y_pred, y_true, N=5)
                all_results[model_name][dataset_name] = acc * 100
                print(f"  {dataset_name:25} | Accuracy: {acc*100:6.2f}%")
        except Exception as e:
            print(f"  {dataset_name:25} | ERROR: {str(e)[:50]}")
            all_results[model_name][dataset_name] = None

# ============================================================================
# PHASE 3: PRINT SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY - Top-5% Accuracy Results")
print("=" * 100)

# Create header
header = f"{'Dataset':30}"
for config in MODEL_CONFIGS:
    header += f" | {config['name']:15}"
print(header)
print("-" * 100)

# Print results for each dataset
for dataset_name in datasets:
    row = f"{dataset_name:30}"
    for config in MODEL_CONFIGS:
        model_name = config["name"]
        acc = all_results[model_name].get(dataset_name)
        if acc is not None:
            row += f" | {acc:14.2f}%"
        else:
            row += f" | {'FAILED':>14}"
    print(row)

# Print model averages
print("-" * 100)
avg_row = f"{'AVERAGE':30}"
for config in MODEL_CONFIGS:
    model_name = config["name"]
    accs = [acc for acc in all_results[model_name].values() if acc is not None]
    if accs:
        avg = sum(accs) / len(accs)
        avg_row += f" | {avg:14.2f}%"
    else:
        avg_row += f" | {'N/A':>14}"
print(avg_row)
print("=" * 100)