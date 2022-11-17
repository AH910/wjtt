import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

import wandb
from data import dataset_attributes
from training import train_model
from utils import get_model, set_seed

# Set seed
set_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters first model
dataset = "celebA"
dataset_attributes = dataset_attributes[dataset]
n_epochs_1 = 1
batch_size_1 = 24
lr_1 = 1e-5
weight_decay_1 = 0.1

# Hyperparameters second model
n_epochs_2 = [30]
batch_size_2 = 24
lr_2 = 1e-5
weight_decay_2 = 0.1
upweight = 50

# Login to wandb
wandb.init(
    project="JTT 2nd model (" + dataset + ")",
    config={
        "Number of epochs 1st model": n_epochs_1,
        "Batch size 1st model": batch_size_1,
        "Learning rate 1st model": lr_1,
        "Weight decay 1st model": weight_decay_1,
        "Number of epochs 2nd model": n_epochs_2,
        "Batch size 2nd model": batch_size_2,
        "Learning rate 2nd model": lr_2,
        "Weight decay 2nd model": weight_decay_2,
    },
)

# Load full dataset
full_dataset = dataset_attributes["class"]()

# Split dataset
train_data, val_data, test_data = full_dataset.split()

# Load error set from first model
error_set = list(
    np.genfromtxt(
        "./results/jtt/"
        + dataset_attributes["dataset"]
        + "/error_sets/"
        + f"nepochs_{n_epochs_1}_"
        + f"lr_{lr_1}_"
        + f"batch_size_{batch_size_1}_"
        + f"wd_{weight_decay_1}.csv",
        delimiter=",",
    )
)
error_set = [int(x) for x in error_set]

# Add the samples, that were incorrectly classified by the 1st model, (upweight-1) times
# to the training data, so that every sample that was correctly classified is in the
# training data once and every sample in the error set is in the training data
# (upweight) times
upweighted_samples = Subset(full_dataset, list(error_set) * upweight)
train_data = ConcatDataset([train_data, upweighted_samples])

# Dataloaders
loader_kwargs = {
    "batch_size": batch_size_2,
    "num_workers": 4,
    "pin_memory": True,
}
train_dataloader = DataLoader(train_data, shuffle=True, **loader_kwargs)
val_dataloader = DataLoader(val_data, **loader_kwargs)
test_dataloader = DataLoader(test_data, **loader_kwargs)

# Initialize model
model = get_model(dataset_attributes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_2,
    momentum=0.9,
    weight_decay=weight_decay_2,
)

train_model(
    model,
    n_epochs_2,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    lr_2,
    batch_size_2,
    weight_decay_2,
    dataset_attributes,
)
