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


def get_weights(probs, alpha, func):
    weights = []

    if func == "JTT":
        for p in probs:
            if p < 0.5:
                w = alpha + 1
            else:
                w = 1

            weights.append(int(w))

    elif func == "DRO1":
        losses = [-np.log(p) for p in probs]
        sum_exp = (1 / len(losses)) * (sum(np.exp(-loss / alpha) for loss in losses))

        for k in range(len(probs)):
            w = np.exp(-losses[k] / alpha) / sum_exp
            weights.append(int(max(1, 100 * w)))
            breakpoint()

    elif func == "DRO2":
        loss = [-np.log(p) for p in probs]
        avg_loss = sum(loss) / len(loss)
        std_loss = np.sqrt(
            sum((loss[k] - avg_loss) ** 2 for k in range(len(loss))) / len(loss)
        )

        for k in range(len(probs)):
            w = np.sqrt(2 * alpha) * ((loss[k] - avg_loss) / std_loss)
            weights.append(int(max(1, 100 * (1 + w))))

    elif func == "CVar":
        alpha, CVar_beta = alpha[0], alpha[1] / 100
        sorted_indices = np.argsort(probs)
        M = int(CVar_beta * len(probs))
        for k in range(len(probs)):
            if k in sorted_indices[: M + 1]:
                weights.append(alpha)
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
