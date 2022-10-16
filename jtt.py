import torch
from waterbird_prep import WBDataset 
from torch.utils.data import DataLoader



full_dataset = WBDataset(data_dir="./data/waterbird_complete95_forest2water2", 
    metadata_csv_name="metadata.csv")