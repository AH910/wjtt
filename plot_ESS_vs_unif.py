import matplotlib.pyplot as plt
import numpy as np

from data import dataset_attributes
from utils import get_weights

# Script for generating Figure 6
# Takes about 12min

# Set font for plot Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"
# Minus sign in CM font
plt.rcParams["axes.unicode_minus"] = False

# Returns the effective sample size for a set of weights.
def get_eff_sample_size(weights):
    return (sum(weights) ** 2) / sum(w**2 for w in weights)


# Which functions to compare
weighting = ["JTT", "DRO2", "w1", "w2"]

# Hyperparameter for different methods
alphas = {
    "JTT": np.linspace(0, 20, 101),
    "DRO2": np.linspace(0, 15, 101),
    "w1": np.linspace(0, 4, 101),
    "w2": np.linspace(0, 2, 101),
}
rho = 2

# Initialize dict. for ESS stats
ESS = {
    "waterbird": {
        "JTT": [],
        "DRO2": [],
        "w1": [],
        "w2": [],
    },
    "celebA": {
        "JTT": [],
        "DRO2": [],
        "w1": [],
        "w2": [],
    },
}

# Initialize dictionary for uniformity stats
uniformity = {
    "waterbird": {
        "JTT": [],
        "DRO2": [],
        "w1": [],
        "w2": [],
    },
    "celebA": {
        "JTT": [],
        "DRO2": [],
        "w1": [],
        "w2": [],
    },
}

for dataset in ["waterbird", "celebA"]:
    # Load dataset to get group labels
    dataset_attrs = dataset_attributes[dataset]
    full_dataset = dataset_attrs["class"]()
    train_data, val_data, test_data = full_dataset.split()
    groups = [train_data[k][2] for k in range(len(train_data))]
    n_groups = dataset_attrs["n_groups"]

    # Load error_set
    error_set = "./error_sets/" + dataset + "/err_set.csv"
    probs = list(np.genfromtxt(error_set, delimiter=","))

    # Go through all functions and parameters and calculate ESS and uniformity
    for func in weighting:
        for alpha in alphas[func]:
            # get weights and calculate rel. weight of groups
            weights = get_weights(probs, alpha, rho, func)
            g_weights = [[] for k in range(n_groups)]
            for k in range(len(weights)):
                g_weights[groups[k]].append(weights[k])
            rel_weights = [sum(g_weights[k]) / sum(weights) for k in range(n_groups)]

            # Calculate effective sample size
            ESS[dataset][func].append(np.log(get_eff_sample_size(weights)))

            # Append product of rel. group sizes
            uniformity[dataset][func].append(np.log(np.prod(rel_weights)))

# Make plot
fig, axs = plt.subplots(1, 2)
ylabel = "Log(product)"
xlabel = "Log(ESS)"

# waterbird plot
for func in weighting:
    axs[0].plot(ESS["waterbird"][func], uniformity["waterbird"][func], label=func)
axs[0].title.set_text("waterbird")
axs[0].set_ylabel(ylabel)
axs[0].set_xlabel(xlabel)
axs[0].legend()

# celebA plot
for func in weighting:
    axs[1].plot(ESS["celebA"][func], uniformity["celebA"][func], label=func)
axs[1].title.set_text("celebA")
axs[1].set_xlabel(xlabel)
axs[1].legend()

# Save plot
plt.savefig("ESS_vs_prod.png", dpi=300)
plt.savefig("ESS_vs_prod.eps", dpi=300)
