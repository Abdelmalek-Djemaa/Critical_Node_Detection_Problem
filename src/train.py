import torch
from tqdm import tqdm
from metrics import pairwise_ranking_loss

def train_model(model, data, y_true, epochs=1000, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data, y_true = data.to(device), y_true.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    progress_bar = tqdm(range(epochs), desc="Training Model", dynamic_ncols=True)
    
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        y_pred = model(data)
        loss = pairwise_ranking_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
