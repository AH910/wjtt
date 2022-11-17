import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(dataset_attributes):
    if dataset_attributes["dataset"] == "waterbird":
        model = models.resnet50(pretrained=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, dataset_attributes["n_classes"])
        return model

    elif dataset_attributes["dataset"] == "celebA":
        model = models.resnet50(pretrained=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, dataset_attributes["n_classes"])
        return model
