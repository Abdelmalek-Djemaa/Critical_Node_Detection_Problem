import torch
from tqdm import tqdm
from metrics import pairwise_ranking_loss

def train_model(model, data, y_true, epochs=1000, lr=0.001, batch_size=256):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    data, y_true = data.to(device), y_true.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    progress_bar = tqdm(range(epochs), desc="Training Model", dynamic_ncols=True)
    
    num_nodes = y_true.shape[0]
    
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # For large datasets, compute loss on a random subset to avoid memory issues
        if num_nodes > 1000:
            # Sample a subset of nodes for pairwise ranking loss
            sample_size = min(batch_size, num_nodes)
            indices = torch.randperm(num_nodes)[:sample_size]
            
            # Create a mini-batch data
            mini_data = data.clone()
            mini_data.x = data.x[indices] if hasattr(data.x, '__getitem__') else data.x
            
            y_pred_full = model(data)
            y_pred = y_pred_full[indices]
            y_true_subset = y_true[indices]
        else:
            y_pred = model(data)
            y_true_subset = y_true
        
        loss = pairwise_ranking_loss(y_pred, y_true_subset)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
