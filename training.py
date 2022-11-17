import numpy as np
import pandas as pd
import torch

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, dataloader, criterion, wandb_group, dataset_attributes):
    """
    Evaluates the model. Returns dictionary with group accuracies, worst group accuracy,
    average (over batches) accuracy and total loss of the samples in the dataloader.
    """
    total_loss = 0.0
    total_correct_pred = 0.0
    n_groups = dataset_attributes["n_groups"]
    g_correct_pred = np.zeros(n_groups)
    g_total = np.zeros(n_groups)

    groups = []
    for k in range(n_groups):
        groups.append(dataset_attributes[f"group{k}"])

    if wandb_group == "val":
        print("VALIDATION")
    elif wandb_group == "test":
        print("TESTING")

    for batch_idx, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        data_idx = batch[3]

        # Put model in evaluation mode
        torch.cuda.empty_cache()
        model.eval()

        # Evaluate model
        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y)

        # Update stats
        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_true = y.cpu().numpy()
        total_loss += loss.mean().item()
        total_correct_pred += np.sum(y_pred == y_true)
        for k in range(len(data_idx)):
            g_total[g[k]] += 1
            if y_true[k] == y_pred[k]:
                g_correct_pred[g[k]] += 1
    stats1 = {
        wandb_group
        + "/"
        + f"Group {k} acc."
        + "("
        + groups[k]
        + ")": g_correct_pred[k] / g_total[k]
        for k in range(n_groups)
    }
    stats2 = {
        wandb_group
        + "/"
        + "Worst group acc.": min(
            g_correct_pred[i] / g_total[i] for i in range(n_groups)
        ),
        wandb_group + "/" + "accuracy": total_correct_pred / len(dataloader.dataset),
        wandb_group + "/" + "loss": total_loss,
    }
    print(
        "Worst group accuracy: ",
        min(g_correct_pred[i] / g_total[i] for i in range(n_groups)),
    )
    return {**stats1, **stats2}


def get_error_set(model, dataloader):
    """
    Returns np.array with the (dataset-) indices of all samples in the dataloader
    which are classified incorrectly by the model.
    """
    model.eval()
    error_set = np.array([])

    for batch_idx, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        data_idx = batch[3]

        # Put model in evaluation mode
        torch.cuda.empty_cache()
        model.eval()

        # Evaluate model
        with torch.no_grad():
            outputs = model(x)

        y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        y_true = y.cpu().numpy()

        indices = []
        for k in range(len(data_idx)):
            if y_true[k] != y_pred[k]:
                indices.append(data_idx[k].cpu())
        error_set = np.append(error_set, indices)

    error_set.sort()
    return error_set


def train_model(
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
):
    """
    Trains model on the data in according to train_dataloader, criterion and optimizer
    for n_epochs. Logs training loss/acc., validation loss/acc. and group/worst group
    acc. to wandb and saves error set of the final model as .csv.
    If n_epochs is an array, the error set gets saved for every n in n_epochs.
    """
    if n_epochs is int:
        n_epochs = [n_epochs]

    for epoch in range(max(n_epochs)):
        print(f"\n EPOCH {epoch+1}:\n")
        train_loss = 0.0
        train_correct_pred = 0.0

        # Put model in training mode
        model.train()

        for batch_idx, batch in enumerate(train_dataloader):

            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]

            # Forward pass
            outputs = model(x)
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

        # Logging loss and accuracy to wandb
        train_stats = {
            "train/loss": train_loss / len(train_dataloader),
            "train/accuracy": train_correct_pred / len(train_dataloader.dataset),
        }
        print("TRAINING")
        print(
            f"training loss: {train_loss / len(train_dataloader)}, ",
            f"training accuracy: {train_correct_pred / len(train_dataloader.dataset)}",
        )
        val_stats = eval_model(
            model, val_dataloader, criterion, "val", dataset_attributes
        )
        test_stats = eval_model(
            model, test_dataloader, criterion, "test", dataset_attributes
        )
        stats = {**train_stats, **val_stats, **test_stats}
        wandb.log(stats, step=epoch + 1)

        if epoch + 1 in n_epochs:
            pd.DataFrame(get_error_set(model, train_dataloader)).to_csv(
                "./results/jtt/"
                + dataset_attributes["dataset"]
                + "/error_sets/"
                + f"nepochs_{epoch+1}_"
                + f"lr_{lr}_"
                + f"batch_size_{batch_size}_"
                + f"wd_{weight_decay}.csv",
                header=None,
                index=None,
            )
