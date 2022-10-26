import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import wandb
from waterbird_prep import WBDataset

# Hyperparameters first model
n_epochs_1 = 50
batch_size_1 = 64
lr_1 = 1e-3
weight_decay_1 = 1.0

# Hyperparameters second model
n_epochs_2 = 100
batch_size_2 = 64
lr_2 = 1e-3
weight_decay_2 = 1.0
upweight = 20

# Login to wandb
wandb.init(project="JTT (2nd model)")

# Set seed
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda")

# Load full dataset
full_dataset = WBDataset(
    data_dir="./data/waterbird_complete95_forest2water2",
    metadata_csv_name="metadata.csv",
)

# Split dataset
train_data, val_data, test_data = full_dataset.split()

# Load error set from first model
error_set = np.genfromtxt(
    "./results/jtt/waterbird/error_sets/"
    + f"nepochs_{n_epochs_1}_"
    + f"lr_{lr_1}_"
    + f"batch_size_{batch_size_1}_"
    + f"wd_{weight_decay_1}.csv",
    delimiter=",",
)

# Add the samples, that were incorrectly classified by the 1st model, (upweight-1) times
# to the training data, so that every sample that was correctly classified is in the
# training data once and every sample in the error set is in the training data
# (upweight) times
upweighted_samples = Subset(full_dataset, list(error_set) * (upweight - 1))
train_data = ConcatDataset([train_data, upweighted_samples])

# Split dataset and define dataloaders and sampler for trainloader
train_dataloader = DataLoader(train_data, batch_size=batch_size_2)
val_dataloader = DataLoader(val_data, batch_size=batch_size_2)
test_dataloader = DataLoader(test_data, batch_size=batch_size_2)
