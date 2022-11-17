from celeba_prep import CelebADataset
from civilcomments_prep import CivilCommentsDataset
from multinli_prep import MultiNLIDataset
from waterbird_prep import WBDataset

dataset_attributes = {
    "waterbird": {
        "dataset": "waterbird",
        "class": WBDataset,
        "n_classes": 2,
        "n_groups": 4,
        "group0": "LB on land",
        "group1": "LB on water",
        "group2": "WB on land",
        "group3": "WB on water",
    },
    "celebA": {
        "dataset": "celebA",
        "class": CelebADataset,
        "n_classes": 2,
        "n_groups": 4,
        "group0": "not blond, f",
        "group1": "not blond, m",
        "group2": "blond, f",
        "group3": "blond, m",
    },
    "MultiNLI": {
        "dataset": "MultiNLI",
        "class": MultiNLIDataset,
        "n_classes": 3,
        "n_groups": 6,
        "group0": "contr., no neg.",
        "group1": "contr., neg.",
        "group2": "entailm., no neg.",
        "group3": "entailm., neg.",
        "group4": "neutral, no neg.",
        "group5": "neutral, neg.",
    },
    "CivilComments": {
        "dataset": "CivilComments",
        "class": CivilCommentsDataset,
        "n_classes": 2,
        "n_groups": 4,
        "group0": "not toxic, no id.",
        "group1": "not toxic, id.",
        "group2": "toxic, no id.",
        "group3": "toxic, id.",
    },
}
