import os
import torch
import pickle
from tqdm import tqdm
from utils import nx_to_pyg
from models import GNCR
from metrics import pairwise_ranking_loss, top_n_accuracy

def run_pipeline(prefix, model_path, epochs=50, model_kwargs=None):
    if model_kwargs is None: model_kwargs = {}
    
    # Force CPU for training stability on Mac/Complex GNNs
    device = torch.device("cpu")
    
    # Load Data
    data_list = []
    if os.path.exists("train_data"):
        for f in sorted(os.listdir("train_data")):
            if f.startswith(prefix) and f.endswith(".pkl"):
                with open(os.path.join("train_data", f), "rb") as file:
                    entry = pickle.load(file)
                    d = nx_to_pyg(entry["graph"])
                    d.y = torch.tensor(entry["scores"], dtype=torch.float32)
                    data_list.append(d)

    if not data_list: return
    
    model = GNCR(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    pbar = tqdm(range(epochs), desc=f"Training {prefix}")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for data in data_list:
            optimizer.zero_grad()
            data = data.to(device)
            pred = model(data).view(-1)
            loss = pairwise_ranking_loss(pred, data.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/len(data_list))

    torch.save(model.state_dict(), model_path)