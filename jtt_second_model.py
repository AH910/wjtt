import torch
import torchvision.models as models

import wandb

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
wandb.init(project="test")

# Set seed
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda")

# Load first model
model = models.resnet50()
model.load_state_dict(
    torch.load(
        "./models/jtt/first_models/waterbird/resnet50_"
        + f"nepochs_{n_epochs_1}_"
        + f"lr_{lr_1}_"
        + f"batch_size_{batch_size_1}_"
        + f"wd_{weight_decay_1}"
    )
)
model.eval()

# TODO: Load data and upweight the samples incorrectly classified
# by the first model
