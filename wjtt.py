import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import wandb
from data import dataset_attributes
from training import train_model
from utils import get_model, get_weights, set_seed


def run_exp(args):
    # Set seed
    set_seed(args["seed"])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    dataset = args["dataset"]
    n_epochs = args["num_epochs"]
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    weight_decay = args["weight_decay"]

    if type(n_epochs) == int:
        n_epochs = [n_epochs]

    # weighting for second model
    func = args["weight_func"]
    alpha = args["alpha"]
    CVar_beta = args["CVar_beta"]

    #
    id_model = args["id_model"]
    UseWandb = args["UseWandb"]
    dataset_attrs = dataset_attributes[dataset]
    error_set = os.path.join(args["error_set_dir"], args["error_set_name"])

    # Login to wandb
    if id_model and UseWandb:
        wandb.init(
            project="WJTT 1st model (" + dataset + ")",
            config={
                "Number of epochs": max(n_epochs),
                "Batch size": batch_size,
                "Learning rate": lr,
                "Weight decay": weight_decay,
            },
        )
    elif (not id_model) and UseWandb:
        wandb.init(
            project="WJTT 2nd model (" + dataset + ")",
            config={
                "Number of epochs": n_epochs,
                "Batch size": batch_size,
                "Learning rate": lr,
                "Weight decay": weight_decay,
            },
        )

    # Loading full dataset
    full_dataset = dataset_attrs["class"]()

    if id_model:
        # Splitting the full dataset into train, val, and test data according
        # to the split column in metadata.csv
        train_data, val_data, test_data = full_dataset.split()

    else:
        # Splitting the full dataset into (train data is built later), val, and
        # test data according to the split column in metadata.csv
        _, val_data, test_data = full_dataset.split()

        # Load error set from first model
        indices, probs = list(np.genfromtxt(error_set, delimiter=","))
        indices = [int(x) for x in indices]

        # get weights
        if func == "CVar":
            alpha = [alpha, CVar_beta]
        weights = get_weights(probs, alpha, func)

        # get train indices of new training set
        train_indices = []
        for k in range(len(indices)):
            train_indices = train_indices + [indices[k]] * weights[k]

        # training data
        train_data = Subset(full_dataset, list(train_indices))

    # Dataloaders
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_dataloader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_dataloader = DataLoader(val_data, shuffle=True, **loader_kwargs)
    test_dataloader = DataLoader(test_data, **loader_kwargs)

    # Initialize model
    model = get_model(dataset_attrs)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Run training
    train_model(
        model,
        id_model,
        n_epochs,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        criterion,
        lr,
        batch_size,
        weight_decay,
        dataset_attrs,
    )


# Create parser
parser = argparse.ArgumentParser()

# Add dataset to parser
parser.add_argument(
    "--dataset",
    type=str,
    choices=["waterbird", "celebA", "CivilComments", "MultiNLI"],
    required=True,
)

# Add hyperparameters to parser
parser.add_argument("--num_epochs")
parser.add_argument("--batch_size")
parser.add_argument("--learning_rate")
parser.add_argument("--weight_decay")

# Add weighting function
parser.add_argument(
    "--weight_func", choices=["JTT", "DRO2", "CVar", "w1", "w2"], default="JTT"
)
parser.add_argument("--alpha", type=float)
parser.add_argument("--CVar_beta", type=float)

# Other arguments
parser.add_argument(
    "--id_model", default="False", help="If identification model True, otherwise False."
)
parser.add_argument("--seed", default=0)
parser.add_argument(
    "--UseWandb", default="True", help="Logging to Weights&Biases if True."
)
parser.add_argument("--error_set_dir")
parser.add_argument("--error_set_name")

# Parse arguments
args = vars(parser.parse_args())

# Default hyperparameters if not specified
if args["dataset"] == "waterbird":
    if args["num_epochs"] is None:
        args["num_epochs"] = 60
    if args["batch_size"] is None:
        args["batch_size"] = 64
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 1.0
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/waterbird/"
    if args["error_set_name"] is None:
        args["error_set_name"] = "nepochs_60_lr_1e-05_batch_size_64_wd_1.0.csv"


elif args["dataset"] == "celebA":
    if args["num_epochs"] is None:
        args["num_epochs"] = 60
    if args["batch_size"] is None:
        args["batch_size"] = 64
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 1.0
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/celebA/"
    if args["error_set_name"] is None:
        args["error_set_name"] = "nepochs_1_lr_1e-05_batch_size_64_wd_0.1.csv"

elif args["dataset"] == "CivilComments":
    if args["num_epochs"] is None:
        args["num_epochs"] = 60
    if args["batch_size"] is None:
        args["batch_size"] = 64
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 1.0
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/CivilComments/"
    if args["error_set_name"] is None:
        args["error_set_name"] = "nepochs_2_lr_1e-05_batch_size_16_wd_0.01.csv"

elif args["dataset"] == "MultiNLI":
    if args["num_epochs"] is None:
        args["num_epochs"] = 60
    if args["batch_size"] is None:
        args["batch_size"] = 64
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 1.0
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/MultiNLI/"
    if args["error_set_name"] is None:
        args["error_set_name"] = ""

args["id_model"] = args["id_model"].lower() in ["true", "y", "yes"]
args["UseWandb"] = args["UseWandb"].lower() in ["true", "y", "yes"]
run_exp(args)
