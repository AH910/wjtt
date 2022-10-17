import torch
import torchvision.models as models
import wandb
from torch.utils.data import DataLoader

from utils import hinge_loss
from waterbird_prep import WBDataset

# Login to wandb
wandb.login()
wandb.init()

# Set seed
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda")

# Hyperparameters
n_epochs = 1
batch_size = 64
lr = 1e-5
weight_decay = 1.0
model = models.resnet50()

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

# Loss and optimizer
criterion = hinge_loss
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    momentum=0.9,
    weight_decay=weight_decay,
)

# Put model in training mode
model.train()

# Run training
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        data_idx = batch[3]

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print(f"epoch {epoch+1} / {n_epochs} :::::::: loss = {loss.item():.3f}")
            wandb.log({"loss": loss.item()}, step=epoch)
