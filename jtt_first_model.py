import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

import wandb
from waterbird_prep import WBDataset

# Login to wandb
# wandb.init(project="test_run")

# Set seed
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda")

# Hyperparameters
n_classes = 2
n_epochs = [1]
batch_size = 4
lr = 1e-3
weight_decay = 1.0
model = models.resnet50()

d = model.fc.in_features
model.fc = nn.Linear(d, n_classes)

# Loading full waterbird dataset
full_dataset = WBDataset(
    data_dir="./data/waterbird_complete95_forest2water2",
    metadata_csv_name="metadata.csv",
)

# Splitting the full dataset into train, val, and test data according to the split
# column in metadata.csv
train_data, val_data, test_data = full_dataset.split()

# Dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Send model weights to GPU
# model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    momentum=0.9,
    weight_decay=weight_decay,
)

# Run training

for epoch in range(max(n_epochs)):
    # Training
    train_loss = 0.0
    train_correct_pred = 0.0
    error_set = np.array([])

    # Put model in training mode
    # model.train()

    for batch_idx, batch in enumerate(train_dataloader):

        # batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        data_idx = batch[3]

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        # optimizer.zero_grad()
        # loss.mean().backward()
        # optimizer.step()

        # Add loss of this batch to total and add correctly predicted samples to total
        train_loss += loss.mean().item()
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_true = y.cpu().numpy()
        if batch_idx % 100 == 0:
            print(f"data_idx: {data_idx}")
            print(f"outputs: {outputs.detach().numpy()}")
            print(f"y_pred: {y_pred}")
            print(f"y_true: {y_true}")

        train_correct_pred += np.sum(y_pred == y_true)

        if epoch + 1 in n_epochs:
            indices = []
            for k in range(len(data_idx)):
                if y_true[k] != y_pred[k]:
                    indices.append(data_idx[k].cpu())
            error_set = np.append(error_set, indices)
    break
    # Validation
    val_loss = 0.0
    val_correct_pred = 0.0

    for batch_idx, batch in enumerate(val_dataloader):
        # Put model in evaluation mode
        model.eval()

        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        data_idx = batch[3]

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Add loss of this batch to total and add correctly predicted samples to total
        val_loss += loss.mean().item()
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_true = y.cpu().numpy()
        val_correct_pred += np.sum(y_pred == y_true)

    # Logging loss and accuracy to wandb
    wandb.log(
        {
            "training loss": train_loss / len(train_dataloader),
            "training accuracy": train_correct_pred / len(train_dataloader.dataset),
            "validation loss": val_loss / len(val_dataloader),
            "validation accuracy": val_correct_pred / len(val_dataloader.dataset),
        },
        step=epoch + 1,
    )

    if epoch + 1 in n_epochs:
        # Save error set
        error_set.sort()
        pd.DataFrame(error_set).to_csv(
            "./results/jtt/waterbird/error_sets/"
            + f"nepochs_{epoch+1}_"
            + f"lr_{lr}_"
            + f"batch_size_{batch_size}_"
            + f"wd_{weight_decay}.csv",
            header=None,
            index=None,
        )
        # # Save model
        # torch.save(
        #     model.state_dict(),
        #     "./models/jtt/first_models/resnet50_"
        #     + f"nepochs_{epoch+1}_"
        #     + f"lr_{lr}_"
        #     + f"batch_size_{batch_size}_"
        #     + f"wd_{weight_decay}",
        # )
