import torch
import os
from models import GNCR
from pipeline import run_pipeline
from utils import load_graph_dataset, nx_to_pyg, load_criticality_scores
from metrics import top_n_accuracy

# Define comprehensive ablation experiments
# All models use SAGEConv layers (GCN backbone)
ABLATION_CONFIGS = [
    # === COMPONENT ABLATION (SAGEConv always present) ===
    {"name": "GCN Only (Baseline)", "kwargs": {"use_attention": False, "use_jk": False, "regression_depth": 2}},
    {"name": "GCN + JK", "kwargs": {"use_attention": False, "use_jk": True, "regression_depth": 2}},
    {"name": "GCN + Attention", "kwargs": {"use_attention": True, "use_jk": False, "regression_depth": 2}},
    {"name": "GCN + Attention + JK (Full)", "kwargs": {"use_attention": True, "use_jk": True, "regression_depth": 2}},
    
    # === REGRESSION DEPTH TEST (Full Model Only) ===
    {"name": "Full Model (Depth 2)", "kwargs": {"use_attention": True, "use_jk": True, "regression_depth": 2}},
    {"name": "Full Model (Depth 3)", "kwargs": {"use_attention": True, "use_jk": True, "regression_depth": 3}},
    {"name": "Full Model (Depth 4)", "kwargs": {"use_attention": True, "use_jk": True, "regression_depth": 4}},
]

def run_ablation_study():
    os.makedirs("models", exist_ok=True)
    
    # Get all datasets for testing
    data_dir = "data"
    datasets = sorted([f for f in os.listdir(data_dir) if f.endswith(".mtx")])
    
    print("=" * 130)
    print("ABLATION STUDY - SAGEConv Always Present + Regression Depth Analysis")
    print("Training Data: Using 30+ graphs from train_data/pl_*.pkl")
    print("=" * 130)
    
    # Dictionary to store results: {config_name: {dataset: accuracy}}
    all_results = {config["name"]: {} for config in ABLATION_CONFIGS}
    
    # ============================================================================
    # PHASE 1: TRAIN ALL MODELS
    # ============================================================================
    print("\n[PHASE 1] TRAINING ALL MODELS (using train_data folder)")
    print("-" * 130)
    
    for config in ABLATION_CONFIGS:
        model_name = config["name"]
        model_kwargs = config["kwargs"]
        model_path = f"models/ablation_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').lower()}.pth"
        
        if not os.path.exists(model_path):
            print(f"\nTraining [{model_name}]")
            print(f"  Config: {model_kwargs}")
            run_pipeline("pl", model_path, epochs=50, model_kwargs=model_kwargs)
        else:
            print(f"[{model_name}] Already trained (skipping)")
    
    # ============================================================================
    # PHASE 2: TEST ALL MODELS ON ALL DATASETS
    # ============================================================================
    print("\n" + "=" * 130)
    print("[PHASE 2] TESTING ON ALL DATASETS")
    print("=" * 130)
    
    for config in ABLATION_CONFIGS:
        model_name = config["name"]
        model_kwargs = config["kwargs"]
        model_path = f"models/ablation_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').lower()}.pth"
        
        print(f"\n[{model_name}]")
        print("-" * 130)
        
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
    print("\n" + "=" * 130)
    print("SUMMARY - Top-5% Accuracy Results (All models use SAGEConv)")
    print("=" * 130)
    
    # Create header
    header = f"{'Configuration':35}"
    for dataset_name in datasets:
        header += f" | {dataset_name:15}"
    header += " | AVG"
    print(header)
    print("-" * 130)
    
    # Print results for each configuration
    for config in ABLATION_CONFIGS:
        model_name = config["name"]
        row = f"{model_name:35}"
        accs = []
        
        for dataset_name in datasets:
            acc = all_results[model_name].get(dataset_name)
            if acc is not None:
                row += f" | {acc:14.2f}%"
                accs.append(acc)
            else:
                row += f" | {'FAILED':>14}"
        
        if accs:
            avg = sum(accs) / len(accs)
            row += f" | {avg:6.2f}%"
        else:
            row += f" | {'N/A':>6}"
        
        print(row)
    
    print("=" * 130)
    
    # ============================================================================
    # PHASE 4: COMPONENT ANALYSIS
    # ============================================================================
    print("\n" + "=" * 130)
    print("COMPONENT IMPORTANCE ANALYSIS (SAGEConv always present)")
    print("=" * 130)
    
    # Find baselines for comparison
    baseline = None
    for config in ABLATION_CONFIGS:
        if config["name"] == "GCN Only (Baseline)":
            baseline_accs = [acc for acc in all_results[config["name"]].values() if acc is not None]
            baseline = sum(baseline_accs) / len(baseline_accs) if baseline_accs else 0
            print(f"\nBaseline (GCN Only): {baseline:.2f}%")
            break
    
    print("\nComponent Additions (vs Baseline):")
    print("-" * 130)
    
    component_configs = [
        ("GCN + JK", "GCN + JK"),
        ("GCN + Attention", "GCN + Attention"),
        ("GCN + Attention + JK (Full)", "GCN + Attention + JK (Full)"),
    ]
    
    for display_name, config_name in component_configs:
        for cfg in ABLATION_CONFIGS:
            if cfg["name"] == config_name:
                accs = [acc for acc in all_results[config_name].values() if acc is not None]
                avg = sum(accs) / len(accs) if accs else 0
                diff = avg - baseline if baseline else 0
                print(f"{display_name:35} | Avg: {avg:6.2f}% | Δ: {diff:+6.2f}%")
                break
    
    print("\nRegression Depth Impact (Full Model with SAGEConv):")
    print("-" * 130)
    
    depth_configs = [
        ("Full Model (Depth 2)", "Full Model (Depth 2)"),
        ("Full Model (Depth 3)", "Full Model (Depth 3)"),
        ("Full Model (Depth 4)", "Full Model (Depth 4)"),
    ]
    
    depth_results = {}
    for display_name, config_name in depth_configs:
        for cfg in ABLATION_CONFIGS:
            if cfg["name"] == config_name:
                accs = [acc for acc in all_results[config_name].values() if acc is not None]
                avg = sum(accs) / len(accs) if accs else 0
                depth_results[config_name] = avg
                print(f"{display_name:35} | Avg: {avg:6.2f}%")
                break
    
    # Find best depth
    if depth_results:
        best_depth_name = max(depth_results, key=depth_results.get)
        best_depth_acc = depth_results[best_depth_name]
        print(f"\n✓ Best Regression Depth: {best_depth_name} with {best_depth_acc:.2f}%")
    
    print("=" * 130)

if __name__ == "__main__":
    run_ablation_study()