import torch


def hinge_loss(yhat, y):
    """From JTT: The torch loss takes in three arguments so we need to split yhat
    It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    so we need to swap yhat[:, 0] and yhat[:, 1]..."""
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)
