import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from data import dataset_attributes
from training import train_model
from utils import get_model, set_seed

# Set seed
set_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
dataset = "waterbird"
dataset_attributes = dataset_attributes[dataset]
n_epochs = [30, 50, 100, 150]
batch_size = 64
lr = 1e-5
weight_decay = 1.0

# Login to wandb
wandb.init(
    project="JTT 1st model (" + dataset + ")",
    config={
        "Number of epochs": max(n_epochs),
        "Batch size": batch_size,
        "Learning rate": lr,
        "Weight decay": weight_decay,
    },
)

# Loading full dataset
full_dataset = dataset_attributes["class"]()

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

# Initialize model
model = get_model(dataset_attributes)
model = model.to(device)

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
    dataset_attributes,
)
