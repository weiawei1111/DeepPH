import pickle
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from data_split import *
from egnn_model_split_range import DeeppH

device = "cuda:0"
config = {
    'node_input_dim': 21 + 184, # precomputed + updated
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 1,
    'augment_eps': 0.15,
    'task':'pH',
    'batch_size': 16,
    'folds': 5,
    'r':15,
    'num_workers':8,
    "random_seed": 0
}

torch.manual_seed(config["random_seed"])

class RangeR2Loss(nn.Module):
    def forward(self, y_pred_min, y_pred_max, y_true_min, y_true_max):
        is_range_mask = (y_true_max - y_true_min)>=1
        not_range_mask = (y_true_max - y_true_min)<1
        # AVG loss
        y_pred = (y_pred_max+y_pred_min)/2
        below_range = torch.clamp(y_true_min - y_pred, min=0)
        above_range = torch.clamp(y_pred - y_true_max, min=0)
        true_middle_val = (y_true_max+y_true_min)/2
        avg_loss = (below_range ** 2 + above_range ** 2) * not_range_mask
        avg_normalize_term = ((true_middle_val-true_middle_val.mean())**2).sum()
        avg_normalized_loss = avg_loss.sum()/(avg_normalize_term+1e-15)
        # Range loss
        low_error = ((y_pred_min - y_true_min)**2)*is_range_mask
        high_error = ((y_pred_max - y_true_max)**2)*is_range_mask
        low_normalize_term = ((y_true_min-y_true_min.mean())**2).sum()
        high_normalize_term = ((y_true_max-y_true_max.mean())**2).sum()
        range_normalized_loss = (low_error.sum()/(low_normalize_term+1e-15))+(high_error.sum()/(high_normalize_term+1e-15))
        # direction loss
        range_loss = torch.clamp(y_pred_min - y_pred_max, min=0).sum()
        # range-size loss
        range_size_loss = torch.clamp(y_pred_max-y_pred_min, min=0) - torch.clamp(y_true_max-y_true_min, min=0)
        range_size_loss = (range_size_loss.abs()*is_range_mask).sum()

        return avg_normalized_loss + range_normalized_loss + range_loss + range_size_loss*0.1

def get_data():

    print("Loading Testing Set")

    with open("/data/Optimum_pH/data/new_test_value.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_dataset = ProteinPHValueGraphDataset(test_data, radius=config['r'])
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        drop_last=False, 
        num_workers=config['num_workers'], 
        prefetch_factor=2
    )

    print("Loading Training Set")
    with open("/data/Optimum_pH/data/new_train_value.pkl", "rb") as f:
        train_data = pickle.load(f)

    train_dataset = ProteinPHValueGraphDataset(train_data, radius=config['r'], split="training")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=False, 
        num_workers=config['num_workers'], 
        prefetch_factor=2
    )
    
    print("Loading Validation Set")
    val_dataset = ProteinPHValueGraphDataset(train_data, radius=config['r'], split="validation")
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=False, 
        num_workers=config['num_workers'], 
        prefetch_factor=2
    )
    return test_dataloader, train_dataloader, val_dataloader

def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, epochs):
    all_train_loss = []
    all_val_loss = []
    model.train()
    for epoch in range(epochs):
        # Training
        model, loss = train_step(model, criterion, optimizer, train_dataloader, epoch, epochs)
        print(f"Epoch {epoch+1}, Loss: {loss}")
        all_train_loss.append(loss / len(train_dataloader))
        # Validation
        val_loss = validate_step(model, criterion, val_dataloader)
        print("Val Loss:", val_loss)
        all_val_loss.append(val_loss)
    print("Training complete.")
    return model, all_train_loss, all_val_loss

def train_step(model, criterion, optimizer, train_dataloader, epoch, epochs):
    running_loss = 0.0
    for data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        data = data.to(device)
        output = model.forward(
            data.X, data.structure_feat, data.seq_feat,
            data.edge_index, data.batch
        )
        pred_ph_min = output[:,0]
        pred_ph_max = output[:,1]
        loss = criterion(pred_ph_min, pred_ph_max, data.ph_min, data.ph_max)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    del data, output, loss
    return model, running_loss/len(train_dataloader)

def validate_step(model, criterion, val_dataloader):
    with torch.no_grad():
        running_loss = 0.0
        for data in tqdm(val_dataloader):
            data = data.to(device)
            output = model.forward(
                data.X, data.structure_feat, data.seq_feat,
                data.edge_index, data.batch
            )
            pred_ph_min = output[:,0]
            pred_ph_max = output[:,1]
            loss = criterion(pred_ph_min, pred_ph_max, data.ph_min, data.ph_max)
            running_loss += loss.item()
    del data, output, loss
    return running_loss / len(val_dataloader)

def evaluate_model(model, test_dataloader):
    model.eval()
    all_mae = []
    all_mae_low = []
    all_mae_high = []

    all_output = []
    all_output_low = []
    all_output_high = []

    all_true = []
    all_true_low = []
    all_true_high = []
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Evaluating"):
            data = data.to(device)
            output = model.forward(
                data.X, data.structure_feat, data.seq_feat,
                data.edge_index, data.batch
            )
            is_range_mask = (data.ph_max - data.ph_min)>=1
            not_range_mask = (data.ph_max - data.ph_min)<1
            mae = ((data.ph_min + data.ph_max)/2)-output.mean(axis=1)
            mae_low = (data.ph_min - output[:, 0]).abs()[is_range_mask]
            mae_high = (data.ph_max - output[:, 1]).abs()[is_range_mask]
            all_mae.append(mae.abs()[not_range_mask])
            all_mae_low.append(mae_low)
            all_mae_high.append(mae_high)
            all_output.append(output.mean(axis=1)[not_range_mask])
            all_output_low.append(output[:,0][is_range_mask])
            all_output_high.append(output[:,1][is_range_mask])
            all_true.append((data.ph_min+data.ph_max)[not_range_mask]/2)
            all_true_low.append(data.ph_min[is_range_mask])
            all_true_high.append(data.ph_max[is_range_mask])
    print("Not-Range:")
    all_output = torch.concat(all_output).cpu()
    all_true = torch.concat(all_true).cpu()
    all_mae = torch.concat(all_mae)
    mae = float(all_mae.mean())
    rmse = float((all_mae**2).mean()**0.5)
    r_2 = 1-((all_mae**2).sum()/((all_true-all_true.mean())**2).sum())
    pearson_cc = pearsonr(all_output, all_true)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r_2:.4f}")
    print(f"Pearson:", pearson_cc.statistic, pearson_cc.pvalue)
    print()

    print("Range-min:")
    all_output_low = torch.concat(all_output_low).cpu()
    all_true_low = torch.concat(all_true_low).cpu()
    all_mae_low = torch.concat(all_mae_low)
    mae_low = float(all_mae_low.mean())
    rmse_low = float((all_mae_low**2).mean()**0.5)
    r_2_low = 1-((all_mae_low**2).sum()/((all_true_low-all_true_low.mean())**2).sum())
    pearson_cc_low = pearsonr(all_output_low, all_true_low)
    print(f"MAE: {mae_low:.4f}")
    print(f"RMSE: {rmse_low:.4f}")
    print(f"R2: {r_2_low:.4f}")
    print(f"Pearson:", pearson_cc_low.statistic, pearson_cc_low.pvalue)
    print()
    
    print("Range-max")
    all_output_high = torch.concat(all_output_high).cpu()
    all_true_high = torch.concat(all_true_high).cpu()
    all_mae_high = torch.concat(all_mae_high)
    mae_high = float(all_mae_high.mean())
    rmse_high = float((all_mae_high**2).mean()**0.5)
    r_2_high = 1-((all_mae_high**2).sum()/((all_true_high-all_true_high.mean())**2).sum())
    pearson_cc_high = pearsonr(all_output_high, all_true_high)
    print(f"MAE: {mae_high:.4f}")
    print(f"RMSE: {rmse_high:.4f}")
    print(f"R2: {r_2_high:.4f}")
    print(f"Pearson:", pearson_cc_high.statistic, pearson_cc_high.pvalue)
    print()

    # Compute intersection
    inter_left = torch.max(all_output_low, all_true_low)
    inter_right = torch.min(all_output_high, all_true_high)
    intersection = torch.clamp(inter_right - inter_left, min=0)

    # Compute union
    union_left = torch.min(all_output_low, all_true_low)
    union_right = torch.max(all_output_high, all_true_high)
    union = union_right - union_left

    # Compute IoU
    iou = intersection / union  # shape: (size,)
    print("IoU:", iou.mean().item())

if __name__ == "__main__":
    test_dataloader, train_dataloader, val_dataloader = get_data()
    train_epochs = 3
    model = DeeppH(
        config['node_input_dim'], 
        config['edge_input_dim'], 
        config['hidden_dim'], 
        config['layer'], 
        config['augment_eps'], 
        config['task'], 
        device
    ).to(device)

    criterion = RangeR2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    model, train_loss, val_loss = train_model(
        model, 
        criterion, optimizer, 
        train_dataloader, val_dataloader, train_epochs
    )
    log = pd.DataFrame([train_loss, val_loss], index=["train", "val"]).T
    log.to_csv("test_model.csv")
    torch.save(model, "test_model.pt")
    # model = torch.load("test_model_r2_split_aa_range.pt")
    evaluate_model(model, test_dataloader)
    """
    test_model_r2_split_aa_range: 
        pH loss only
    
    test_model_r2_split_aa_range2:
        pH loss + (ph_min-ph_max) loss
    
    test_model_r2_split_aa_range3:
        pH loss + {ph_min-ph_max} loss + 0.1*{ph-range loss}
    """