import matplotlib.pyplot as plt
import numpy as np

from data import dataset_attributes
from utils import get_weights

# Script to generate Figure 4

# Set font for plot Computer Modern (LaTex font)
plt.rcParams["font.family"] = "cmr10"

# Set measure for "balancedness" of groups
def unif(rel_weights):
    return np.prod(rel_weights)


# Set weighting parameters for JTT and WJTT (DRO k=2 weights)
param = {
    "waterbird": {"alpha": 8, "rho": 2, "JTT_upweight": 100},
    "celebA": {"alpha": 8, "rho": 1, "JTT_upweight": 50},
    "CivilComments": {"alpha": 3, "rho": 0.5, "JTT_upweight": 6},
    "MultiNLI": {"alpha": 3, "rho": 0.5, "JTT_upweight": 6},
}

# Initialize uniformity dictionary
uniformity = {
    "waterbird": {"org": 0, "JTT": 0, "WJTT": 0},
    "celebA": {"org": 0, "JTT": 0, "WJTT": 0},
    "CivilComments": {"org": 0, "JTT": 0, "WJTT": 0},
    "MultiNLI": {"org": 0, "JTT": 0, "WJTT": 0},
}

for dataset in ["waterbird", "celebA", "CivilComments", "MultiNLI"]:
    # Set parameters
    alpha = param[dataset]["alpha"]
    rho = param[dataset]["rho"]
    JTT_upweight = param[dataset]["JTT_upweight"]

    # Load dataset and split to get training data (for group labels)
    dataset_attrs = dataset_attributes[dataset]
    full_dataset = dataset_attrs["class"]()
    train_data, val_data, test_data = full_dataset.split()

    # Get group labels for the training set
    groups = [train_data[k][2] for k in range(len(train_data))]

    # Load error set
    error_set = "./error_sets/" + dataset + "/err_set.csv"
    probs = list(np.genfromtxt(error_set, delimiter=","))
    # For MultiNLI error_set is 2 arrays (probabilities is the first one)
    if dataset == "MultiNLI":
        misclassified = probs[1]
        probs = probs[0]

    # Get weights
    wjtt_w = get_weights(probs, alpha, rho, "DRO2")
    if dataset == "MultiNLI":
        jtt_w = [int(alpha + 1) if m == 1 else 1 for m in misclassified]
    else:
        jtt_w = get_weights(probs, JTT_upweight, 0, "JTT")

    # Initialize lists for group counts
    n_g = dataset_attrs["n_groups"]  # Number of groups in dataset
    groups_org = [0 for k in range(n_g)]  # Original dataset
    groups_jtt = [0 for k in range(n_g)]  # JTT dataset
    groups_wjtt = [0 for k in range(n_g)]  # WJTT dataset

    for k in range(len(groups)):
        g = groups[k]  # get group label
        # Add weight to group counts
        groups_org[g] += 1
        groups_jtt[g] += jtt_w[k]
        groups_wjtt[g] += wjtt_w[k]

    # Lists with relative group sizes
    rel_groups_org = [g / sum(groups_org) for g in groups_org]
    rel_groups_jtt = [g / sum(groups_jtt) for g in groups_jtt]
    rel_groups_wjtt = [g / sum(groups_wjtt) for g in groups_wjtt]

    # Set maximal "uniformity" and calculate percentages
    max_unif = unif([1 / n_g] * n_g)
    uniformity[dataset]["org"] = 100 * unif(rel_groups_org) / max_unif
    uniformity[dataset]["JTT"] = 100 * unif(rel_groups_jtt) / max_unif
    uniformity[dataset]["WJTT"] = 100 * unif(rel_groups_wjtt) / max_unif

    print(f"Done with {dataset}.")

# Make plot
labels = ["org.", "JTT", "WJTT"]
dataset = ["waterbird", "celebA", "CivilComments", "MultiNLI"]
fig, axs = plt.subplots(1, 4)
for k in range(4):
    unif = [uniformity[dataset[k]][x] for x in ["org", "JTT", "WJTT"]]
    barlist = axs[k].bar(labels, unif, color="orange", edgecolor="black")
    barlist[0].set_color("royalblue")
    barlist[0].set_edgecolor(color="black")
    if k == 3:  # MultiNLI
        axs[k].set_ylim([0, 1])
    else:
        axs[k].set_ylim([0, 100])
    axs[k].title.set_text(dataset[k])

for k in range(4):
    axs[k].spines["top"].set_visible(False)
    axs[k].spines["right"].set_visible(False)
    if k == 0 or k == 3:
        continue
    axs[k].get_yaxis().set_visible(False)
    axs[k].spines["left"].set_visible(False)

plt.subplots_adjust(hspace=0.8)
plt.tight_layout()

# Save plot
plt.savefig("unif_groups_comparison.png", dpi=300)
plt.savefig("unif_groups_comparison.eps", dpi=300)
