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

    # Set hyperparameters
    dataset = args["dataset"]
    n_epochs = args["num_epochs"]
    n_epochs = [int(x) for x in n_epochs.strip("][").split(",")]
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    weight_decay = args["weight_decay"]

    # Set variables for weighting (second model)
    func = args["weight_func"]
    alpha = args["alpha"]
    CVar_beta = args["CVar_beta"]

    # Set other variables
    id_model = args["id_model"]  # ==True for 1st model and ==False for 2nd model
    UseWandb = args["UseWandb"]  # logging to Weights&Biases if ==True
    dataset_attrs = dataset_attributes[dataset]
    error_set = "./" + os.path.join(args["error_set_dir"], args["error_set_name"])

    # Login to Weights&Biases
    # 1st model
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
    # 2nd model
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

    # Setting train, validation and test datasets
    # 1st model
    if id_model:
        # Splitting the full dataset into train, val, and test data according
        # to the split column in metadata.csv
        train_data, val_data, test_data = full_dataset.split()

    # 2nd model
    else:
        # Splitting the full dataset into (train data is built later), val, and
        # test data according to the split column in metadata.csv
        temp_train_data, val_data, test_data = full_dataset.split()

        # Load error set from first model
        probs = list(np.genfromtxt(error_set, delimiter=","))
        indices = [temp_train_data[k][3] for k in range(len(temp_train_data))]
        breakpoint()
        # get weights
        if func == "CVar":
            alpha = [alpha, CVar_beta]
        weights = get_weights(probs, alpha, func)
        breakpoint()
        # get train indices of new training set
        train_indices = []
        for k in range(len(indices)):
            train_indices = train_indices + [indices[k]] * weights[k]

        # training data
        train_data = Subset(full_dataset, list(train_indices))
    breakpoint()
    # Setting train, validation and test dataloader
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

    # Setting loss function
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
        UseWandb,
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
parser.add_argument("--batch_size", type=int)
parser.add_argument("--learning_rate")
parser.add_argument("--weight_decay")

# Add weighting function and parameters
parser.add_argument(
    "--weight_func", choices=["JTT", "DRO2", "CVar", "w1", "w2"], default="JTT"
)
parser.add_argument("--alpha", type=float)
parser.add_argument("--CVar_beta", type=float)

# Other arguments
parser.add_argument(
    "--id_model", default="False", help="If identification model True, otherwise False."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--UseWandb", default="True", help="Logging to Weights&Biases if True."
)
parser.add_argument("--error_set_dir")
parser.add_argument("--error_set_name")

# Parse arguments and convert to dictionary
args = vars(parser.parse_args())

# Default hyperparameters if not specified
# waterbird
if args["dataset"] == "waterbird":
    if args["num_epochs"] is None:
        args["num_epochs"] = "60"
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

# celebA
elif args["dataset"] == "celebA":
    if args["num_epochs"] is None:
        args["num_epochs"] = "5"
    if args["batch_size"] is None:
        args["batch_size"] = 64
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 0.1
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/celebA/"
    if args["error_set_name"] is None:
        args["error_set_name"] = "nepochs_1_lr_1e-05_batch_size_64_wd_0.1.csv"

# CivilComments
elif args["dataset"] == "CivilComments":
    if args["num_epochs"] is None:
        args["num_epochs"] = "5"
    if args["batch_size"] is None:
        args["batch_size"] = 16
    if args["learning_rate"] is None:
        args["learning_rate"] = 1e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 0.01
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/CivilComments/"
    if args["error_set_name"] is None:
        args["error_set_name"] = "nepochs_2_lr_1e-05_batch_size_16_wd_0.01.csv"

# MultiNLI
elif args["dataset"] == "MultiNLI":
    if args["num_epochs"] is None:
        args["num_epochs"] = "4"
    if args["batch_size"] is None:
        args["batch_size"] = 32
    if args["learning_rate"] is None:
        args["learning_rate"] = 2e-5
    if args["weight_decay"] is None:
        args["weight_decay"] = 0
    if args["error_set_dir"] is None:
        args["error_set_dir"] = "./error_sets/MultiNLI/"
    if args["error_set_name"] is None:
        args["error_set_name"] = ""

# Convert strings id_model and UseWandb to bools
args["id_model"] = args["id_model"].lower() in ["true", "y", "yes"]
args["UseWandb"] = args["UseWandb"].lower() in ["true", "y", "yes"]

# Run experiment with arguments from parser
run_exp(args)
