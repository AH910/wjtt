import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax

import wandb
from utils import argmax, argmin

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, dataloader, criterion, wandb_group, dataset_attrs):
    """
    Evaluates the model. Returns dictionary with group accuracies, worst group accuracy,
    average (over batches) accuracy and total loss of the samples in the dataloader.
    Prints worst-group accuracy to console
    """

    # Initialize variables and arrays for the stats
    n_groups = dataset_attrs["n_groups"]  # Number of groups in the dataset
    total_loss = 0.0  # Sum of loss over all samples in the dataset
    total_correct_pred = 0.0  # Num of correctly classified samples
    g_correct_pred = np.zeros(n_groups)  # Num of correctly classified samples by group
    g_total = np.zeros(n_groups)  # Total num of samples per group

    # Array with names of the groups
    groups = []
    for k in range(n_groups):
        groups.append(dataset_attrs[f"group{k}"])

    if wandb_group == "val":
        print("VALIDATION")
    elif wandb_group == "test":
        print("TESTING")

    # Iterate through dataloader
    for batch_idx, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)
        x = batch[0]  # images
        y = batch[1]  # labels
        g = batch[2]  # groups
        data_idx = batch[3]  # indices

        # Empty cache and put model in evaluation mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.eval()

        # Evaluate model
        with torch.no_grad():

            # Get outputs for CivilComments dataset
            if dataset_attrs["dataset"] == "CivilComments":
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[
                    1
                ]  # [1] returns logits

            # Get outputs for all the other datasets
            else:
                outputs = model(x)

            # Apply criterion to get losses for this batch
            loss = criterion(outputs, y)

        # Update stats
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)  # predicted label
        y_true = y.cpu().numpy()  # True label
        total_loss += loss.mean().item()
        total_correct_pred += np.sum(y_pred == y_true)

        # Update g_total and g_correct_pred
        for k in range(len(data_idx)):
            g_total[g[k]] += 1
            if y_true[k] == y_pred[k]:
                g_correct_pred[g[k]] += 1

    # Dictionary with group accuracies
    stats1 = {
        wandb_group
        + "/"
        + f"Group {k} acc. "
        + "("
        + groups[k]
        + ")": g_correct_pred[k] / g_total[k]
        for k in range(n_groups)
    }

    # Dictionary with worst-group accuracy, accuracy and loss
    stats2 = {
        wandb_group
        + "/"
        + "Worst group acc.": min(
            g_correct_pred[i] / g_total[i] for i in range(n_groups)
        ),
        wandb_group + "/" + "accuracy": total_correct_pred / len(dataloader.dataset),
        wandb_group + "/" + "loss": total_loss,
    }

    # Print worst-group accuracy
    print(
        "Worst-group accuracy: ",
        "{:.4f}".format(min(g_correct_pred[i] / g_total[i] for i in range(n_groups))),
    )

    # Return dictionary with stats
    return {**stats1, **stats2}


def get_error_set(model, dataloader, dataset_attrs):
    """
    Returns np.array with the probabilities the model gives for each sample of
    it belonging to the true class. (Sorted by sample index.)
    """
    # Initialize error set, first array = indices, second array = probabilities
    error_set = [[], []]
    # Initialize softmax function to get probabilities from cross-entropy loss
    s = Softmax(dim=0)

    # Iterate through dataloader
    for batch_idx, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)
        x = batch[0]  # images
        y = batch[1]  # labels
        data_idx = batch[3]  # indices

        # Put model in evaluation mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.eval()

        # Evaluate model
        with torch.no_grad():

            # Get outputs for CivilComments dataset
            if dataset_attrs["dataset"] == "CivilComments":
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[
                    1
                ]  # [1] returns logits

            # Get outputs for all the other datasets
            else:
                outputs = model(x)

        # Get true label and the probability p given by model for true label
        y_true = y.cpu().numpy()
        p = s(outputs).detach().cpu().numpy()[np.array(range(len(y_true))), y_true]

        # Append indices and probabilities to error_set[0] and error_set[1]
        for k in range(len(data_idx)):
            error_set[0].append(data_idx[k].cpu())
            error_set[1].append(p[k])

    # Sort array of probabilities acc. to indices
    error_set = np.array(error_set)  # Convert to np.array
    ind = np.argsort(error_set[0, :])  # Get indices of sorted array
    error_set = error_set[:, ind]  # Sort array

    # Return probabilities
    return error_set[1]


def train_model(
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
):
    """
    Trains model on the data according to train_dataloader, criterion and optimizer
    for n_epochs. Logs training loss/acc., validation loss/acc. and group/worst group
    acc. to wandb and saves error set of the final model as .csv.
    If n_epochs is an array, the error set gets saved for every n in n_epochs.
    """
    # Step for wandb, so it starts at 1 instead of 0
    step = 1

    # Initialize dictionary with arrays to record some stats during the training
    # which will be used for model selection (early stopping)
    stat_arrays = {
        "epoch": [],
        "val_loss": [],
        "val_avg_acc": [],
        "val_wga": [],
        "test_loss": [],
        "test_avg_acc": [],
        "test_wga": [],
    }

    # Set optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    # How often to log and save error set during epoch:
    if dataset_attrs["dataset"] == "waterbird":
        logging_x_times = 0
    elif dataset_attrs["dataset"] == "celebA":
        logging_x_times = 4
    elif dataset_attrs["dataset"] == "CivilComments":
        logging_x_times = 4
    elif dataset_attrs["dataset"] == "MultiNLI":
        logging_x_times = 4

    # Batch indices at which to log
    log_at_batch_id = [
        int(k * len(train_dataloader) / logging_x_times)
        for k in range(1, logging_x_times)
    ]

    # Start training
    for epoch in range(max(n_epochs)):
        # Initialize training loss and number of correctly predicted samples
        train_loss = 0.0
        train_correct_pred = 0.0

        # Put model in training mode
        model.train()

        # Iterate through dataloader
        for batch_idx, batch in enumerate(train_dataloader):

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch = tuple(t.to(device) for t in batch)
            x = batch[0]  # images
            y = batch[1]  # labels

            # Forward pass
            # Get outputs for CivilComments
            if dataset_attrs["dataset"] == "CivilComments":
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[
                    1
                ]  # [1] returns logits

            # Get outputs for all the other datasets
            else:
                outputs = model(x)

            # Apply criterion to outputs
            loss = criterion(outputs, y)

            # Backward pass
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            # Add loss of this batch to total and add correctly predicted
            # samples to total
            train_loss += loss.mean().item()
            y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            y_true = y.cpu().numpy()
            train_correct_pred += np.sum(y_pred == y_true)

            # Logging during epoch:
            if batch_idx + 1 in log_at_batch_id:

                # How many epochs have passed, e.g. 1.25
                ep = (
                    epoch + (log_at_batch_id.index(batch_idx + 1) + 1) / logging_x_times
                )

                # Logging and printing stats (during epoch)
                train_stats = {
                    "train/loss": train_loss / (batch_idx + 1),
                    "train/accuracy": train_correct_pred
                    / (batch_size * (batch_idx + 1)),
                }
                # Print stats to console
                print(f"EPOCH: {ep}")
                print("TRAINING")
                print("training loss: ", "{:.4f}".format(train_loss / (batch_idx + 1)))
                print(
                    "training accuracy: ",
                    "{:.4f}".format(
                        train_correct_pred / (batch_size * (batch_idx + 1))
                    ),
                )
                print()

                # Testing and validation
                val_stats = eval_model(
                    model, val_dataloader, criterion, "val", dataset_attrs
                )
                test_stats = eval_model(
                    model, test_dataloader, criterion, "test", dataset_attrs
                )
                stats = {**train_stats, **val_stats, **test_stats}

                # Log to Weights&Biases if UseWandb==True
                if UseWandb:
                    wandb.log(stats, step=step)
                    step += 1

                # Update stats in dictionary (for early stopping)
                stat_arrays["epoch"].append(ep)
                stat_arrays["val_loss"].append(stats["val/loss"])
                stat_arrays["val_avg_acc"].append(stats["val/accuracy"])
                stat_arrays["val_wga"].append(stats["val/Worst group acc."])
                stat_arrays["test_loss"].append(stats["test/loss"])
                stat_arrays["test_avg_acc"].append(stats["test/accuracy"])
                stat_arrays["test_wga"].append(stats["test/Worst group acc."])

                # Save error_set as .csv file
                if epoch + 1 in n_epochs and id_model:
                    pd.DataFrame(
                        get_error_set(model, train_dataloader, dataset_attrs)
                    ).to_csv(
                        "./error_sets/"
                        + dataset_attrs["dataset"]
                        + f"/nepochs_{ep}_"
                        + f"lr_{lr}_"
                        + f"batch_size_{batch_size}_"
                        + f"wd_{weight_decay}.csv",
                        header=None,
                        index=None,
                    )

        # Logging and printing stats (after epoch)

        # Training stats
        train_stats = {
            "train/loss": train_loss / len(train_dataloader),
            "train/accuracy": train_correct_pred / len(train_dataloader.dataset),
        }

        # Printing training stats
        print(f"EPOCH: {epoch+1}")
        print("TRAINING")
        print("training loss: ", "{:.4f}".format(train_loss / len(train_dataloader)))
        print(
            "training accuracy: ",
            "{:.4f}".format(train_correct_pred / len(train_dataloader.dataset)),
        )
        print()

        # Validation and testing
        val_stats = eval_model(model, val_dataloader, criterion, "val", dataset_attrs)
        test_stats = eval_model(
            model, test_dataloader, criterion, "test", dataset_attrs
        )
        stats = {**train_stats, **val_stats, **test_stats}

        # Log to Weights&Biases if UseWandb==True
        if UseWandb:
            wandb.log(stats, step=step)
            step += 1

        # Update stats in dictionary (for early stopping)
        stat_arrays["epoch"].append(epoch + 1)
        stat_arrays["val_loss"].append(stats["val/loss"])
        stat_arrays["val_avg_acc"].append(stats["val/accuracy"])
        stat_arrays["val_wga"].append(stats["val/Worst group acc."])
        stat_arrays["test_loss"].append(stats["test/loss"])
        stat_arrays["test_avg_acc"].append(stats["test/accuracy"])
        stat_arrays["test_wga"].append(stats["test/Worst group acc."])

        # Save error_set as .csv file
        if epoch + 1 in n_epochs and id_model:
            pd.DataFrame(get_error_set(model, train_dataloader, dataset_attrs)).to_csv(
                "./error_sets/"
                + dataset_attrs["dataset"]
                + f"/nepochs_{epoch+1}_"
                + f"lr_{lr}_"
                + f"batch_size_{batch_size}_"
                + f"wd_{weight_decay}.csv",
                header=None,
                index=None,
            )

    # Print stats for 2nd model
    if not id_model:
        print("\n")
        print("---------------------------------------------------")
        print("--------Stats achieved with early stopping:--------")
        print("---------------------------------------------------")
        print("\n")

        # Early stopping according to average val loss (not using group labels)
        ind = argmin(stat_arrays["val_loss"])
        max_wga_epoch = stat_arrays["epoch"][ind]
        test_wga = stat_arrays["test_wga"][ind]
        test_avg_acc = stat_arrays["test_avg_acc"][ind]
        print("Early stopping according to average validation loss:")
        print(f"Stopping after {max_wga_epoch} epochs.")
        print("Worst-group acc. on test set: ", "{:.2f}".format(test_wga * 100), "%")
        print("Average acc. on test set: ", "{:.2f}".format(test_avg_acc * 100), "%")
        print("\n")

        # Early stopping according to average val accuracy (not using group labels)
        ind = argmax(stat_arrays["val_avg_acc"])
        max_wga_epoch = stat_arrays["epoch"][ind]
        test_wga = stat_arrays["test_wga"][ind]
        test_avg_acc = stat_arrays["test_avg_acc"][ind]
        print("Early stopping according to average validation accuracy:")
        print(f"Stopping after {max_wga_epoch} epochs.")
        print("Worst-group acc. on test set: ", "{:.2f}".format(test_wga * 100), "%")
        print("Average acc. on test set: ", "{:.2f}".format(test_avg_acc * 100), "%")
        print("\n")

        # Early stopping according to val worst-group acc. (using group labels)
        ind = argmax(stat_arrays["val_wga"])
        max_wga_epoch = stat_arrays["epoch"][ind]
        test_wga = stat_arrays["test_wga"][ind]
        test_avg_acc = stat_arrays["test_avg_acc"][ind]
        print("Early stopping according to average validation worst-group acc.:")
        print(f"Stopping after {max_wga_epoch} epochs.")
        print(
            "Worst-group acc. on test set: ",
            "{:.2f}".format(
                test_wga * 100,
            ),
            " %",
        )
        print("Average acc. on test set: ", "{:.2f}".format(test_avg_acc * 100), " %")
        print("\n")
