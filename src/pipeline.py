import os
import torch
import pickle
from torch_geometric.data import Batch
from utils import nx_to_pyg
from models import GNCR
from train import train_model
from metrics import top_n_accuracy

DATA_DIR = "train_data"

def load_saved_graphs(prefix):
    data_list, y_list = [], []

    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.startswith(prefix) and filename.endswith(".pkl"):
            with open(os.path.join(DATA_DIR, filename), "rb") as f:
                entry = pickle.load(f)
                G, scores = entry["graph"], entry["scores"]
                y = torch.tensor(scores, dtype=torch.float)
                data = nx_to_pyg(G,y)
                data_list.append(data)
                y_list.append(y)

    batch = Batch.from_data_list(data_list)
    y_all = torch.cat(y_list, dim=0)
    return batch, y_all

hidden_dim = 32

def run_pipeline(prefix, model_path, epochs=1000):
    data, y_true = load_saved_graphs(prefix)
    model = GNCR(hidden_dim)
    train_model(model, data, y_true, epochs)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.eval()
    y_pred = model(data).detach()
    acc = top_n_accuracy(y_pred, y_true, N=5)
    print(f"Top-5% Accuracy: {acc * 100:.2f}%")

def main():
    os.makedirs("models", exist_ok=True)

    print("Training on saved Power Law graphs...")
    run_pipeline("pl", "models/model_SAGE_pl.pth")

    print("Training on saved Power Law Cluster graphs...")
    run_pipeline("plc", "models/model_SAGE_plc.pth")

if __name__ == "__main__":
    main()
