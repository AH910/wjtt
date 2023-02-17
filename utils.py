import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertForSequenceClassification


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(dataset_attributes):
    # Import modules
    if dataset_attributes["dataset"] == "CivilComments":
        from transformers import BertForSequenceClassification
    elif dataset_attributes["dataset"] == "MultiNLI":
        from pytorch_transformers import (BertConfig,
                                          BertForSequenceClassification)

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

    elif dataset_attributes["dataset"] == "CivilComments":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        return model

    elif dataset_attributes["dataset"] == "MultiNLI":
        config_class = BertConfig
        model_class = BertForSequenceClassification

        config = config_class.from_pretrained(
            "bert-base-uncased", num_labels=3, finetuning_task="mnli"
        )
        model = model_class.from_pretrained(
            "bert-base-uncased", from_tf=False, config=config
        )
        return model


def get_weights(probs, alpha, rho, func):
    weights = []

    if func == "JTT":
        for p in probs:
            if p < 0.5:
                w = alpha + 1
            else:
                w = 1

            weights.append(int(w))

    elif func == "DRO2":
        # If probs[k]==0, set it to 0.00001 to avoid log(0)
        loss = [-np.log(p) if p != 0 else -np.log(0.00001) for p in probs]
        avg_loss = sum(loss) / len(loss)
        std_loss = np.sqrt(
            sum((loss[k] - avg_loss) ** 2 for k in range(len(loss))) / len(loss)
        )

        for k in range(len(probs)):

            w = np.sqrt(2 * rho) * ((loss[k] - avg_loss) / std_loss)
            w = min(400, alpha * (1 + w))  # Upper bound for weights = 400
            weights.append(int(max(1, w)))  # Lower bound = 1

    elif func == "CVar":
        alpha, CVar_beta = alpha[0], alpha[1] / 100
        sorted_indices = np.argsort(probs)
        M = int(CVar_beta * len(probs))
        for k in range(len(probs)):
            if k in sorted_indices[: M + 1]:
                weights.append(int(alpha))
            else:
                weights.append(1)

    elif func == "w1":
        for p in probs:
            w = (alpha + (1 - p)) / (alpha + p * (1 - p))

            weights.append(int(w))

    elif func == "w2":
        for p in probs:
            w = (alpha * (1 - p)) / (p**2) + 1

            weights.append(min(400, int(w)))

    else:
        print(f"Function {func} not implemented.")

    return weights


# Returns index of the max/min value of arr (if the max/min is reached
# multiple times, we take the last occurence).
def argmax(arr):
    ind, max_num = 0, arr[0]
    for k in range(1, len(arr)):
        if arr[k] >= max_num:
            ind, max_num = k, arr[k]
    return ind


def argmin(arr):
    ind, min_num = 0, arr[0]
    for k in range(1, len(arr)):
        if arr[k] <= min_num:
            ind, min_num = k, arr[k]
    return ind
