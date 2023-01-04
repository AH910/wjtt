import matplotlib.pyplot as plt
import numpy as np

from data import dataset_attributes
from utils import get_weights


def get_eff_sample_size(weights):
    return (sum(weights) ** 2) / sum(w**2 for w in weights)


dataset = "celebA"
dataset_attributes = dataset_attributes[dataset]
n_groups = dataset_attributes["n_groups"]

# Log(min(rel. sum of weights)) or Log(prod(rel. sum of weights))
m_or_p = "p"

#
weighting = ["JTT", "DRO2", "w1", "w2"]
alphas = {
    "JTT": [25, 50, 75, 100, 125, 150, 175],
    "DRO2": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100],
    "w1": [0.5, 0.1, 0.01, 0.001, 0.0001],
    "w2": [4, 6, 8, 10, 12, 14],
}
weighting = ["JTT", "DRO2", "w1", "w2"]

# Load data and error set
error_set = {
    "waterbird": "./error_sets/waterbird/nepochs_60_lr_1e-05_batch_size_64_wd_1.0.csv",
    "celebA": "./error_sets/celebA/nepochs_1_lr_1e-05_batch_size_64_wd_0.1.csv",
    "CivilComments": "./error_sets/CivilComments/nepochs_2_lr_1e-05_batch_size_16_wd_0.01.csv",
    "MultiNLI": "./error_sets/MultiNLI/",
}

indices, probs = list(
    np.genfromtxt(
        error_set[dataset],
        delimiter=",",
    )
)
indices = [int(x) for x in indices]
full_dataset = dataset_attributes["class"]()
groups = full_dataset.group_array[indices]

ESS = {
    "JTT": [],
    "DRO2": [],
    "w1": [],
    "w2": [],
}

uniformity = {
    "JTT": [],
    "DRO2": [],
    "w1": [],
    "w2": [],
}

#
for func in weighting:
    for alpha in alphas[func]:
        # get weights and calculate rel. weight of groups
        weights = get_weights(probs, alpha, func)
        g_weights = [[] for k in range(n_groups)]
        for k in range(len(weights)):
            g_weights[groups[k]].append(weights[k])
        rel_weights = [sum(g_weights[k]) / sum(weights) for k in range(n_groups)]

        # Calculate effective sample size
        ESS[func].append(np.log(get_eff_sample_size(weights)))

        # Calculate uniformity
        if m_or_p == "m":
            uniformity[func].append(np.log(min(rel_weights)))
        else:
            uniformity[func].append(np.log(np.prod(rel_weights)))


# Labels of plot
if m_or_p == "m":
    ylabel = "Log(min(rel. sum of weights))"
elif m_or_p == "p":
    ylabel = "Log(prod(rel. sum of weights))"

for func in weighting:
    plt.plot(ESS[func], uniformity[func], label=func)
plt.title(dataset)
plt.xlabel("Log(eff_sample_size)")
plt.ylabel(ylabel)
plt.legend()
plt.savefig(f"{dataset}_" + m_or_p, dpi=300)
