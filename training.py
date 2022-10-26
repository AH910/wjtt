import numpy as np
import pandas as pd
import torch

import wandb

device = torch.device("cuda")


def eval_model(model, dataloader, criterion):
    """
    Evaluates the model. Returns dictionary with group accuracies, worst group accuracy,
    average (over batches) accuracy and total loss of the samples in the dataloader.
    """
    total_loss = 0.0
    total_correct_pred = 0.0
    g_correct_pred = np.zeros(4)
    g_total = np.zeros(4)

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

    stats = {
        "Val: Group 0 acc. (LB on land)": g_correct_pred[0] / g_total[0],
        "Val: Group 1 acc. (LB on water)": g_correct_pred[1] / g_total[1],
        "Val: Group 2 acc. (WB on land)": g_correct_pred[2] / g_total[2],
        "Val: Group 3 acc. (WB on water)": g_correct_pred[3] / g_total[3],
        "Val: Worst group acc.": min(
            g_correct_pred[i] / g_total[i] for i in [0, 1, 2, 3]
        ),
        "validation accuracy": total_correct_pred / len(dataloader.dataset),
        "validation loss": total_loss,
    }

    return stats


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
    criterion,
    optimizer,
    lr,
    batch_size,
    weight_decay,
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
        stats = eval_model(model, val_dataloader, criterion)
        stats["training loss"] = train_loss / len(train_dataloader)
        stats["training accuracy"] = train_correct_pred / len(train_dataloader.dataset)
        wandb.log(stats, step=epoch + 1)

        if epoch + 1 in n_epochs:
            pd.DataFrame(get_error_set(model, train_dataloader)).to_csv(
                "./results/jtt/waterbird/error_sets/"
                + f"nepochs_{epoch+1}_"
                + f"lr_{lr}_"
                + f"batch_size_{batch_size}_"
                + f"wd_{weight_decay}.csv",
                header=None,
                index=None,
            )
