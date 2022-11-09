import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

import wandb
from training import train_model
from waterbird_prep import WBDataset

# Set seed
np.random.seed(0)
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda")

# Hyperparameters
n_classes = 2
n_epochs = [30, 50, 100, 150]
batch_size = 64
lr = 1e-5
weight_decay = 1.0

# Login to wandb
wandb.init(
    project="test_run",
    config={
        "Number of epochs": max(n_epochs),
        "Batch size": batch_size,
        "Learning rate": lr,
        "Weight decay": weight_decay,
    },
)

# Initialize model
model = models.resnet50(pretrained=True)
d = model.fc.in_features
model.fc = nn.Linear(d, n_classes)
model = model.to(device)

# Loading full waterbird dataset
full_dataset = WBDataset(
    data_dir="./data/waterbird_complete95_forest2water2",
    metadata_csv_name="metadata.csv",
)

# Splitting the full dataset into train, val, and test data according to the split
# column in metadata.csv
train_data, val_data, test_data = full_dataset.split()

# Dataloaders
loader_kwargs = {
    "batch_size": batch_size,
    "num_workers": 4,
    "pin_memory": True,
}
train_dataloader = DataLoader(train_data, shuffle=True, **loader_kwargs)
val_dataloader = DataLoader(val_data, **loader_kwargs)
test_dataloader = DataLoader(test_data, **loader_kwargs)


# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    momentum=0.9,
    weight_decay=weight_decay,
)

# Run training
train_model(
    model,
    n_epochs,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    criterion,
    optimizer,
    lr,
    batch_size,
    weight_decay,
)
